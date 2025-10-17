from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
print("Starting Anomaly API Service...")

app = FastAPI(title="Anomaly Detection API", version="1.0.0")
print("FastAPI app initialized.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Allow both ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request bodies
class StatusUpdate(BaseModel):
    status: str

def get_db_conn():
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=os.environ.get('POSTGRES_PORT', 5432),
        dbname=os.environ.get('POSTGRES_DB', 'anomalydb'),
        user=os.environ.get('POSTGRES_USER', 'postgres'),
        password=os.environ.get('POSTGRES_PASSWORD', 'postgres')
    )


@app.get("/api/dashboard/overview")
def get_dashboard_overview():
    """Get overview metrics for the dashboard"""
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Total anomalies
        cur.execute("SELECT COUNT(*) as total FROM anomalies")
        total = cur.fetchone()['total']
        
        # Open anomalies
        cur.execute("SELECT COUNT(*) as open FROM anomalies WHERE status = 'Open'")
        open_count = cur.fetchone()['open']
        
        # Acknowledged anomalies
        cur.execute("SELECT COUNT(*) as acknowledged FROM anomalies WHERE status = 'Acknowledged'")
        acknowledged_count = cur.fetchone()['acknowledged']
        
        # Resolved anomalies
        cur.execute("SELECT COUNT(*) as resolved FROM anomalies WHERE status = 'Resolved'")
        resolved_count = cur.fetchone()['resolved']
        
        # Severity breakdown
        cur.execute("""
            SELECT severity, COUNT(*) as count 
            FROM anomalies 
            GROUP BY severity
        """)
        severity_counts = cur.fetchall()
        
        # Recent anomalies (last 24 hours)
        cur.execute("""
            SELECT COUNT(*) as recent 
            FROM anomalies 
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """)
        recent_count = cur.fetchone()['recent']
        
        severity_distribution = {row['severity']: row['count'] for row in severity_counts}
        
        return {
            "total_anomalies": total,
            "open_anomalies": open_count,
            "acknowledged_anomalies": acknowledged_count,
            "resolved_anomalies": resolved_count,
            "recent_anomalies": recent_count,
            "severity_distribution": severity_distribution
        }
        
    except Exception as e:
        print(f"Error in get_dashboard_overview: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        cur.close()
        conn.close()

@app.get("/api/dashboard/timeline")
def get_timeline(days: int = Query(7, description="Number of days to look back")):
    """Get timeline data for anomaly counts over time"""
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Build a continuous series of dates for the requested range and left join counts
        # Calculate start_date and end_date in Python and pass as parameters to avoid interval string issues
        from datetime import date as _date
        end_date = _date.today()
        start_date = end_date - timedelta(days=days - 1)

        cur.execute("""
            SELECT gs::date AS date, COALESCE(c.count, 0) AS count
            FROM generate_series(%s::date, %s::date, '1 day') gs
            LEFT JOIN (
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM anomalies
                WHERE timestamp >= %s::date
                GROUP BY DATE(timestamp)
            ) c ON gs::date = c.date
            ORDER BY date
        """, (start_date, end_date, start_date))

        timeline_data = cur.fetchall()

        # Convert to list of dictionaries with string dates
        result = [
            {"date": str(row['date']), "count": row['count']} for row in timeline_data
        ]

        return result
        
    except Exception as e:
        print(f"Error in get_timeline: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        cur.close()
        conn.close()

@app.get("/api/dashboard/distributions")
def get_distributions():
    """Get distribution data for charts"""
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Status distribution
        cur.execute("""
            SELECT status, COUNT(*) as count 
            FROM anomalies 
            GROUP BY status
        """)
        status_dist = cur.fetchall()
        
        # Enterprise distribution (top 10)
        cur.execute("""
            SELECT enterprise_id, COUNT(*) as count 
            FROM anomalies 
            GROUP BY enterprise_id
            ORDER BY count DESC
            LIMIT 10
        """)
        enterprise_dist = cur.fetchall()
        
        status_distribution = {row['status']: row['count'] for row in status_dist}
        enterprise_distribution = [
            {"enterprise_id": row['enterprise_id'], "count": row['count']} 
            for row in enterprise_dist
        ]
        
        return {
            "status_distribution": status_distribution,
            "enterprise_distribution": enterprise_distribution
        }
        
    except Exception as e:
        print(f"Error in get_distributions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        cur.close()
        conn.close()

@app.get("/api/anomalies/all")
def get_all_anomalies(
    status: Optional[str] = Query(None, description="Filter by status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    enterprise_id: Optional[str] = Query(None, description="Filter by enterprise ID (use 'all' for no filter)"),
    limit: int = Query(100, description="Number of results to return"),
    offset: int = Query(0, description="Number of results to skip")
):
    """Get all anomalies with filtering and pagination"""
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Build dynamic query
        query = "SELECT * FROM anomalies WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = %s"
            params.append(status)
        
        if severity:
            query += " AND severity = %s"
            params.append(severity)
            
        if enterprise_id and enterprise_id != 'all':
            query += " AND enterprise_id = %s"
            params.append(enterprise_id)
        
        query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cur.execute(query, params)
        anomalies = cur.fetchall()
        
        # Get total count for pagination
        count_query = "SELECT COUNT(*) as total FROM anomalies WHERE 1=1"
        count_params = []
        
        if status:
            count_query += " AND status = %s"
            count_params.append(status)
        
        if severity:
            count_query += " AND severity = %s"
            count_params.append(severity)
            
        if enterprise_id and enterprise_id != 'all':
            count_query += " AND enterprise_id = %s"
            count_params.append(enterprise_id)
        
        cur.execute(count_query, count_params)
        total = cur.fetchone()['total']
        
        # Convert timestamps to strings for JSON serialization
        anomalies_list = []
        for anomaly in anomalies:
            anomaly_dict = dict(anomaly)
            if anomaly_dict['timestamp']:
                anomaly_dict['timestamp'] = str(anomaly_dict['timestamp'])
            anomalies_list.append(anomaly_dict)
        
        return {
            "anomalies": anomalies_list,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
        
    except Exception as e:
        print(f"Error in get_all_anomalies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        cur.close()
        conn.close()

@app.get("/api/anomalies/search")
def search_anomalies(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, description="Number of results to return")
):
    """Search anomalies by resource name, enterprise ID, or resource ID"""
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cur.execute("""
            SELECT * FROM anomalies 
            WHERE resource_name ILIKE %s 
               OR enterprise_id ILIKE %s 
               OR resource_id ILIKE %s
            ORDER BY timestamp DESC 
            LIMIT %s
        """, (f'%{q}%', f'%{q}%', f'%{q}%', limit))
        
        results = cur.fetchall()
        
        # Convert timestamps to strings for JSON serialization
        results_list = []
        for result in results:
            result_dict = dict(result)
            if result_dict['timestamp']:
                result_dict['timestamp'] = str(result_dict['timestamp'])
            results_list.append(result_dict)
        
        return results_list
        
    except Exception as e:
        print(f"Error in search_anomalies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        cur.close()
        conn.close()


@app.put("/api/anomalies/{anomaly_id}/status")
def update_anomaly_status(anomaly_id: int, status_update: StatusUpdate):
    """Update the status of an anomaly"""
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Validate status value
        new_status = status_update.status
        valid_statuses = ['Open', 'Acknowledged', 'Resolved']
        
        if new_status not in valid_statuses:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )
        
        # Update the anomaly status
        cur.execute(
            "UPDATE anomalies SET status = %s WHERE id = %s RETURNING *",
            (new_status, anomaly_id)
        )
        
        updated_anomaly = cur.fetchone()
        
        if not updated_anomaly:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        
        conn.commit()
        
        # Convert timestamp to string for JSON serialization
        result = dict(updated_anomaly)
        if result['timestamp']:
            result['timestamp'] = str(result['timestamp'])
        
        print(f"Successfully updated anomaly {anomaly_id} status to {new_status}")
        return result
        
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        print(f"Error in update_anomaly_status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        cur.close()
        conn.close()


@app.get("/api/health")
def health_check():
    """Health check endpoint for monitoring"""
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        conn.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "disconnected",
            "error": str(e)
        }