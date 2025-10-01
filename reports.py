import json
import logging
import requests
import xmltodict
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from xml.parsers.expat import ExpatError
from sqlalchemy.orm import Session, joinedload
from datetime import date, timedelta, datetime
from pathlib import Path
 
# Import user model, auth functions, and db session getter
from .. import models, auth
from ..database import get_db

from sqlalchemy import func
# Load credentials from environment variables
API_URL = "https://go.xlence.com/api/"

router = APIRouter()

# Pydantic models for response validation and serialization
class ReportRow(BaseModel):
    Day: Optional[str] = None
    Year: Optional[str] = None
    Month: Optional[str] = None
    Brand: Optional[str] = None
    TrackingCode: Optional[str] = None
    Language: Optional[str] = None
    Type: Optional[str] = None
    Size: Optional[str] = None
    Name: Optional[str] = None
    Impressions: int
    Visitors: int
    Registrations: int
    FTD: int
    Commission: float

class ReportResponse(BaseModel):
    reports: List[ReportRow]

class DashboardStats(BaseModel):
    total_users: int
    sessions_in_period: int
    messages_in_period: int
    knowledge_files_count: int
    active_users_in_period: int
    period_from: str
    period_to: str
    chart_labels: List[str]
    daily_sessions: List[int]
    daily_messages: List[int]
    knowledge_file_type_labels: List[str]
    knowledge_file_type_counts: List[int]
    recent_sessions: List["RecentSession"]

class RecentSession(BaseModel):
    id: int
    title: str
    username: str
    created_at: datetime

    class Config:
        from_attributes = True

def _apply_filters(rows: List, filters_json: Optional[str]) -> List:
    """Applies include/exclude filters to a list of report rows."""
    if not filters_json:
        return rows

    try:
        parsed_filters = json.loads(filters_json)
        if not parsed_filters:
            return rows

        include_filters = [f for f in parsed_filters if f.get('type') == 'include']
        exclude_filters = [f for f in parsed_filters if f.get('type') == 'exclude']

        def check_condition(row_value, operator, filter_value):
            """Checks a single value against an operator and filter value."""
            row_str = str(row_value or '').lower()
            filter_str = str(filter_value or '').lower()

            if operator == 'is': return row_str == filter_str
            if operator == 'is_not': return row_str != filter_str
            if operator == 'contains': return filter_str in row_str
            if operator == 'does_not_contain': return filter_str not in row_str
            if operator == 'starts_with': return row_str.startswith(filter_str)
            if operator == 'ends_with': return row_str.endswith(filter_str)
            return False

        def row_matches(row):
            """Determines if a row should be included based on all filters."""
            passes_includes = all(check_condition(row.get(f.get('column')), f.get('operator'), f.get('value')) for f in include_filters)
            passes_excludes = not any(check_condition(row.get(f.get('column')), f.get('operator'), f.get('value')) for f in exclude_filters)
            return passes_includes and passes_excludes

        return [row for row in rows if row_matches(row)]
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid filters JSON format.")

def _apply_grouping(rows: List, date_group: Optional[str], properties: Optional[List[str]]) -> List:
    """Applies multi-dimensional grouping and aggregation to report rows."""
    group_by_keys = []
    if date_group and date_group in ['Year', 'Month', 'Day']:
        group_by_keys.append(date_group)
    if properties:
        group_by_keys.extend(properties)

    if not group_by_keys:
        return rows

    aggregated_data = {}
    for row in rows:
        key_parts = []
        day_str = row.get('Day')
        if date_group == 'Year': key_parts.append(day_str[:4] if day_str else 'N/A')
        elif date_group == 'Month': key_parts.append(day_str[:7] if day_str else 'N/A')
        elif date_group == 'Day': key_parts.append(day_str if day_str else 'N/A')

        for prop in (properties or []):
            key_parts.append(row.get(prop, 'N/A'))
        
        composite_key = tuple(key_parts)

        if composite_key not in aggregated_data:
            new_group = {"Impressions": 0, "Visitors": 0, "Registrations": 0, "FTD": 0, "Commission": 0.0}
            
            key_part_index = 0
            if date_group and date_group in ['Year', 'Month', 'Day']:
                new_group[date_group] = key_parts[key_part_index]
                key_part_index += 1
            
            for prop in (properties or []):
                new_group[prop] = key_parts[key_part_index]
                key_part_index += 1

            aggregated_data[composite_key] = new_group
        
        aggregated_data[composite_key]["Impressions"] += int(row.get("Impressions", 0))
        aggregated_data[composite_key]["Visitors"] += int(row.get("Visitors", 0))
        aggregated_data[composite_key]["Registrations"] += int(row.get("Registrations", 0))
        aggregated_data[composite_key]["FTD"] += int(row.get("FTD", 0))
        aggregated_data[composite_key]["Commission"] += float(row.get("Commission", 0.0))

    return list(aggregated_data.values())


@router.get("/reports", response_model=ReportResponse)
async def get_media_reports(
    user: models.User = Depends(auth.get_current_user_for_api),
    db: Session = Depends(get_db),
    fromdate: str = Query(..., description="Start date in YYYY-MM-DD format"),
    todate: str = Query(..., description="End date in YYYY-MM-DD format"),
    filters: Optional[str] = Query(None, description="JSON string of filter conditions"),
    date_group: Optional[str] = Query(None, description="Date grouping: Year, Month, or Day"),
    properties: Optional[str] = Query(None, description="Comma-separated properties to group by")
):
    """
    Fetches media reports from the external xlence API using user-specific credentials,
    parses the XML, and returns the data as JSON.
    Includes optional multi-dimensional grouping for data aggregation.
    """
    # The user object from auth includes the profile relationship.
    # Accessing user.profile will lazy-load it using the db session.
    if not user.profile:
        # This case should ideally not happen for a logged-in user
        raise HTTPException(status_code=404, detail="User profile not found.")

    affiliate_id = user.profile.xelence_affiliateid
    api_key = user.profile.xelence_x_api_key

    if not affiliate_id or not api_key:
        raise HTTPException(
            status_code=400,
            detail="Xelence API credentials are not set. Please update them in your settings page."
        )

    selected_properties = properties.split(',') if properties else []

    headers = {'affiliateid': affiliate_id, 'x-api-key': api_key}
    params = {
        'command': 'mediareport',
        'fromdate': fromdate,
        'todate': todate,
        'Brand': '0', # Always fetch all brands to filter dynamically on our end
        'TrackingCode': '1' if 'TrackingCode' in selected_properties else '0',
        'Language': '1' if 'Language' in selected_properties else '0',
        'Type': '1' if 'Type' in selected_properties else '0',
        'Size': '1' if 'Size' in selected_properties else '0',
        'Name': '1' if 'Name' in selected_properties else '0',
    }

    logging.info(
        "Making request to Xelence API with AffiliateID: %s and API Key: %s",
        affiliate_id,
        api_key,
    )

    try:
        response = requests.get(API_URL, headers=headers, params=params, timeout=15)
        response.raise_for_status()

        # If the response is empty, there's no data to parse.
        if not response.content:
            return {"reports": []}

        data_dict = xmltodict.parse(response.content)
        rows = data_dict.get('ResultSet', {}).get('row', [])

        if not isinstance(rows, list):
            rows = [rows]

        processed_rows = []
        optional_columns = ['Brand', 'TrackingCode', 'Language', 'Type', 'Size', 'Name']
        for row in rows:
            if not row.get('Day'):  # Skip rows without a Day, as seen in sample data
                continue
            # Handle optional columns that might be returned as nil from the API
            for col in optional_columns:
                if isinstance(row.get(col), dict) and row[col].get('@xsi:nil') == '1':
                    row[col] = None
            processed_rows.append(row)

        # Apply filtering and grouping using the refactored helper functions
        filtered_rows = _apply_filters(processed_rows, filters)
        grouped_rows = _apply_grouping(filtered_rows, date_group, selected_properties)

        return {"reports": grouped_rows}
    except requests.exceptions.RequestException as e:
        # Check for specific auth error from the external API
        if e.response and e.response.status_code in [401, 403]:
             raise HTTPException(status_code=401, detail="Authentication failed with Xelence API. Please check your credentials in settings.")
        raise HTTPException(status_code=502, detail=f"Failed to fetch data from external API: {e}")
    except ExpatError:
        # This happens if the response is not valid XML, e.g., an HTML error page.
        logging.error(
            "Xelence API returned non-XML response. Status: %s, Body: %s. Used AffiliateID: %s, API Key: %s",
            response.status_code,
            response.text,
            affiliate_id,
            api_key,
        )
        raise HTTPException(
            status_code=502,
            detail="The external reporting service returned an invalid response (not XML). This may be a temporary issue. Please try again later."
        )
    except Exception as e:
        logging.exception("An unexpected error occurred during report processing.")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during report processing: {e}")

@router.get("/dashboard-stats", response_model=DashboardStats)
async def get_dashboard_stats(
    user: models.User = Depends(auth.get_current_user_for_api),
    db: Session = Depends(get_db),
    days: int = Query(30, description="Number of past days to include in stats.", ge=1, le=365)
):
    """
    Fetches and aggregates key statistics from the local system for the main dashboard.
    """
    try:
        # 1. Calculate date range
        todate = date.today()
        fromdate_obj = todate - timedelta(days=days - 1)
        # We need datetime objects for DB queries against DateTime columns
        from_datetime = datetime.combine(fromdate_obj, datetime.min.time())

        # 2. Get cumulative stats
        total_users = db.query(models.User).count()
        active_users_query = db.query(models.ChatSession.user_id).filter(
            models.ChatSession.created_at >= from_datetime
        ).distinct()
        active_users_in_period = active_users_query.count()

        # 3. Get daily stats for the chart
        date_range = [fromdate_obj + timedelta(days=x) for x in range(days)]
        chart_labels = [d.strftime('%Y-%m-%d') for d in date_range]

        # Query for daily sessions
        daily_sessions_query = db.query(
            func.date(models.ChatSession.created_at),
            func.count(models.ChatSession.id)
        ).filter(
            models.ChatSession.created_at >= from_datetime
        ).group_by(
            func.date(models.ChatSession.created_at)
        ).all()
        sessions_dict = {str(d): count for d, count in daily_sessions_query}
        daily_sessions_data = [sessions_dict.get(label, 0) for label in chart_labels]

        # Query for daily messages
        daily_messages_query = db.query(
            func.date(models.ChatMessage.created_at),
            func.count(models.ChatMessage.id)
        ).filter(
            models.ChatMessage.created_at >= from_datetime
        ).group_by(
            func.date(models.ChatMessage.created_at)
        ).all()
        messages_dict = {str(d): count for d, count in daily_messages_query}
        daily_messages_data = [messages_dict.get(label, 0) for label in chart_labels]

        # 4. Get total stats for the period (sum of daily stats)
        sessions_in_period = sum(daily_sessions_data)
        messages_in_period = sum(daily_messages_data)

        # 5. Get knowledge file count (this is not time-bound)
        data_dir = Path("data")
        file_type_counts = {}
        if data_dir.exists():
            supported_extensions = ['.pdf', '.docx', '.txt']
            for p in data_dir.iterdir():
                ext = p.suffix.lower()
                if ext in supported_extensions:
                    label = ext.replace('.', '').upper()
                    file_type_counts[label] = file_type_counts.get(label, 0) + 1

        knowledge_files_count = sum(file_type_counts.values())
        knowledge_file_type_labels = list(file_type_counts.keys())
        knowledge_file_type_counts_data = list(file_type_counts.values())

        # 6. Get recent sessions
        recent_sessions_from_db = db.query(models.ChatSession).options(
            joinedload(models.ChatSession.user)
        ).order_by(models.ChatSession.created_at.desc()).limit(5).all()

        # Manually construct RecentSession objects to handle the nested username.
        # Pydantic's from_attributes can't automatically resolve session.user.username.
        recent_sessions_for_pydantic = []
        for session in recent_sessions_from_db:
            recent_sessions_for_pydantic.append(
                RecentSession(
                    id=session.id,
                    title=session.title,
                    username=session.user.username if session.user else "Unknown User",
                    created_at=session.created_at
                )
            )

        return DashboardStats(
            total_users=total_users,
            sessions_in_period=sessions_in_period,
            messages_in_period=messages_in_period,
            knowledge_files_count=knowledge_files_count,
            active_users_in_period=active_users_in_period,
            period_from=fromdate_obj.strftime('%Y-%m-%d'),
            period_to=todate.strftime('%Y-%m-%d'),
            chart_labels=chart_labels,
            daily_sessions=daily_sessions_data,
            daily_messages=daily_messages_data,
            knowledge_file_type_labels=knowledge_file_type_labels,
            knowledge_file_type_counts=knowledge_file_type_counts_data,
            recent_sessions=recent_sessions_for_pydantic
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

DashboardStats.update_forward_refs()