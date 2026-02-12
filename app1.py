"""
Phase 2: Real-time Slack Integration for Team Compatibility Predictor
Handles real-time message collection, user resolution, and analytics
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import requests
import time
import os
from datetime import datetime, timedelta
from collections import defaultdict
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================

# Slack Bot Token (OAuth & Permissions → Bot User OAuth Token)
SLACK_BOT_TOKEN = "xoxb-10313122964370-10341631935168-B02UyxaXrHvVqIr4FlZbZSOX"

# MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")
db = client["AISubmodule"]

# Collections
users_collection = db["Users"]
user_teams_collection = db["UserTeams"]
slack_users_collection = db["SlackUsers"]
slack_messages_collection = db["SlackMessages"]
project_slack_channels_collection = db["ProjectSlackChannels"]
team_analytics_collection = db["TeamAnalytics"]

# ============================================================================
# SLACK API FUNCTIONS
# ============================================================================

def get_slack_user_info(user_id):
    """
    Fetch user details from Slack API
    Returns: {name, email, display_name, real_name}
    """
    url = "https://slack.com/api/users.info"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    params = {"user": user_id}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        data = response.json()
        
        if data.get("ok") and "user" in data:
            user = data["user"]
            profile = user.get("profile", {})
            
            return {
                "slack_user_id": user_id,
                "name": profile.get("display_name") or profile.get("real_name") or "Unknown",
                "email": profile.get("email", ""),
                "real_name": profile.get("real_name", ""),
                "display_name": profile.get("display_name", ""),
                "avatar": profile.get("image_72", ""),
                "is_bot": user.get("is_bot", False),
                "last_updated": time.time()
            }
        else:
            print(f"❌ Failed to fetch user {user_id}: {data.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching Slack user {user_id}: {str(e)}")
        return None


def get_slack_channel_info(channel_id):
    """
    Fetch channel details from Slack API
    Returns: {channel_name, channel_id}
    """
    url = "https://slack.com/api/conversations.info"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    params = {"channel": channel_id}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        data = response.json()
        
        if data.get("ok") and "channel" in data:
            channel = data["channel"]
            return {
                "channel_id": channel_id,
                "channel_name": channel.get("name", ""),
                "is_private": channel.get("is_private", False),
                "last_updated": time.time()
            }
        else:
            print(f"❌ Failed to fetch channel {channel_id}: {data.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching channel {channel_id}: {str(e)}")
        return None


def get_available_slack_channels():
    """
    Fetch all Slack channels that the bot has access to
    Filters out channels that are already linked to other teams
    Also filters out default/common workspace channels
    Returns: List of available channels
    """
    url = "https://slack.com/api/conversations.list"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    params = {
        "types": "public_channel,private_channel",
        "exclude_archived": True,
        "limit": 200
    }
    
    # Default channels to exclude (common workspace channels)
    DEFAULT_CHANNELS = {
        'general', 'random', 'social', 'new-channel', 
        'all-compatibleteams', 'announcements', 'watercooler'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        
        if data.get("ok") and "channels" in data:
            all_channels = data["channels"]
            
            # Get all already-linked channel IDs from ProjectSlackChannels collection
            linked_channels = set()
            for link in project_slack_channels_collection.find({}, {"channel_id": 1}):
                linked_channels.add(link["channel_id"])
            
            # Filter out already-linked channels and default channels
            available_channels = []
            for channel in all_channels:
                channel_id = channel["id"]
                channel_name = channel.get("name", "").lower()
                
                # Skip if already linked to another team
                if channel_id in linked_channels:
                    continue
                
                # Skip default/common workspace channels
                if channel_name in DEFAULT_CHANNELS:
                    continue
                
                available_channels.append({
                    "id": channel_id,
                    "name": channel.get("name", ""),
                    "is_private": channel.get("is_private", False),
                    "is_member": channel.get("is_member", False)
                })
            
            return available_channels
        else:
            print(f"❌ Failed to fetch channels: {data.get('error', 'Unknown error')}")
            return []
            
    except Exception as e:
        print(f"❌ Error fetching Slack channels: {str(e)}")
        return []


# ============================================================================
# USER & CHANNEL RESOLUTION WITH CACHING
# ============================================================================

def resolve_slack_user(user_id):
    """
    Resolve Slack user ID to name with intelligent caching
    """
    # Check cache first
    cached_user = slack_users_collection.find_one({"slack_user_id": user_id})
    
    # Use cache if recent (less than 7 days old)
    if cached_user and (time.time() - cached_user.get("last_updated", 0)) < 604800:
        return cached_user["name"]
    
    # Fetch from Slack API
    user_info = get_slack_user_info(user_id)
    
    if user_info:
        # Update or insert cache
        slack_users_collection.update_one(
            {"slack_user_id": user_id},
            {"$set": user_info},
            upsert=True
        )
        return user_info["name"]
    
    # Fallback
    return cached_user["name"] if cached_user else f"User_{user_id[:8]}"


def resolve_channel_to_project(channel_id):
    """
    Map Slack channel to project using ProjectSlackChannels collection
    This is the authoritative source of channel-to-project mappings
    """
    # Look up the channel in ProjectSlackChannels collection
    link = project_slack_channels_collection.find_one({"channel_id": channel_id})
    
    if link:
        return {
            "project_id": link["project_id"],
            "project_title": link["project_id"],  # project_id is actually project_title
            "team_id": link.get("team_id"),
            "lead_username": link["lead_username"]
        }
    
    return None


# ============================================================================
# MESSAGE PROCESSING PIPELINE
# ============================================================================

def process_slack_message(event):
    """
    Main message processing function
    Extracts, resolves, and stores messages with analytics
    """
    user_id = event.get("user")
    channel_id = event.get("channel")
    text = event.get("text", "")
    timestamp = event.get("ts", "")
    
    # Ignore bot messages and system messages
    if event.get("subtype") or not user_id:
        return
    
    print(f"\n📨 Processing message from {user_id} in {channel_id}")
    
    # Resolve user
    user_name = resolve_slack_user(user_id)
    
    # Resolve channel to project
    project_info = resolve_channel_to_project(channel_id)
    
    if not project_info:
        print(f"⚠️  Channel {channel_id} not linked to any project. Ignoring message.")
        return
    
    project_title = project_info["project_title"]
    team_id = project_info.get("team_id")
    lead_username = project_info["lead_username"]
    
    print(f"✅ Message linked to project: {project_title} (Lead: {lead_username})")
    
    # Prepare message document
    message_doc = {
        "slack_user_id": user_id,
        "user_name": user_name,
        "channel_id": channel_id,
        "project_title": project_title,
        "project_id": project_title,
        "team_id": team_id,
        "lead_username": lead_username,
        "message": text,
        "text": text,  # Duplicate for compatibility
        "timestamp": float(timestamp),
        "datetime": datetime.fromtimestamp(float(timestamp)).isoformat(),
        "created_at": time.time(),
        "date": datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d"),
        "time": datetime.fromtimestamp(float(timestamp)).strftime("%H:%M:%S")
    }
    
    # Store message
    slack_messages_collection.insert_one(message_doc)
    print(f"💾 Message stored: {user_name}: {text[:50]}...")
    
    # Update analytics if we have team_id
    if team_id and lead_username:
        update_team_analytics(team_id, lead_username, project_title)


# ============================================================================
# ANALYTICS COMPUTATION
# ============================================================================

def update_team_analytics(team_id, team_owner, project_title):
    """
    Calculate and store real-time analytics for a team
    Analyzes messages from the last 24 hours
    """
    if not team_id:
        return
    
    # Get messages from last 24 hours
    cutoff_time = time.time() - 86400  # 24 hours ago
    
    messages = list(slack_messages_collection.find({
        "team_id": team_id,
        "created_at": {"$gte": cutoff_time}
    }))
    
    if not messages:
        print(f"⚠️  No messages found for team {team_id} in last 24h")
        return
    
    # Calculate metrics
    total_messages = len(messages)
    
    # Unique active users
    unique_users = len(set(msg["user_name"] for msg in messages))
    
    # Top contributors
    user_message_counts = defaultdict(int)
    for msg in messages:
        user_message_counts[msg["user_name"]] += 1
    top_contributors = sorted(user_message_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Peak activity hours
    hour_counts = defaultdict(int)
    for msg in messages:
        hour = datetime.fromtimestamp(msg["timestamp"]).hour
        hour_counts[hour] += 1
    peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Calculate engagement score (messages per active user)
    engagement_score = round(total_messages / unique_users, 2) if unique_users > 0 else 0
    
    # Prepare analytics document
    analytics_doc = {
        "team_id": team_id,
        "team_owner": team_owner,
        "project_title": project_title,
        "period": "last_24h",
        "total_messages": total_messages,
        "active_members": unique_users,
        "engagement_score": engagement_score,
        "peak_activity_hours": [{"hour": h, "messages": c} for h, c in peak_hours],
        "top_contributors": [{"name": n, "messages": c} for n, c in top_contributors],
        "last_updated": time.time(),
        "last_updated_iso": datetime.now().isoformat()
    }
    
    # Update or insert analytics
    team_analytics_collection.update_one(
        {"team_id": team_id, "period": "last_24h"},
        {"$set": analytics_doc},
        upsert=True
    )
    
    print(f"📊 Analytics updated for team {team_id}: {total_messages} messages, {unique_users} active members")


# ============================================================================
# BACKGROUND ANALYTICS UPDATER (RUNS EVERY 5 MINUTES)
# ============================================================================

def scheduled_analytics_update():
    """
    Background job to update analytics for all active teams
    Runs every 5 minutes
    """
    while True:
        try:
            print("\n🔄 Running scheduled analytics update...")
            
            active_teams = user_teams_collection.find({"status": "active"})
            
            for team in active_teams:
                team_id = team.get("id")
                team_owner = team.get("username")
                project_title = team.get("project_title")
                
                if team_id and team_owner and project_title:
                    update_team_analytics(team_id, team_owner, project_title)
            
            print("✅ Scheduled analytics update complete")
            
        except Exception as e:
            print(f"❌ Error in scheduled analytics: {str(e)}")
        
        # Wait 5 minutes
        time.sleep(300)


# ============================================================================
# FLASK WEBHOOK ENDPOINT
# ============================================================================

app = Flask(__name__)
CORS(app)

@app.route('/webhooks/slack', methods=['POST'])
def slack_events():
    """
    Main Slack Events API webhook endpoint
    Handles all incoming Slack events
    """
    data = request.json
    
    # Slack URL verification challenge
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data["challenge"]})
    
    # Process events
    event = data.get("event", {})
    event_type = event.get("type")
    
    print(f"\n🔵 Received event: {event_type}")
    
    # Handle message events
    if event_type == "message" and "subtype" not in event:
        try:
            process_slack_message(event)
        except Exception as e:
            print(f"❌ Error processing message: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Acknowledge receipt immediately (Slack requires response within 3 seconds)
    return jsonify({"ok": True}), 200


# ============================================================================
# API ENDPOINTS FOR FRONTEND
# ============================================================================

@app.route('/api/available-channels', methods=['GET'])
def get_available_channels():
    """
    Get list of Slack channels that are NOT already linked to other teams
    Optionally exclude a specific team's current channel (for re-linking)
    """
    try:
        # Optional: team_id to exclude its current channel from the "already linked" filter
        exclude_team_id = request.args.get('exclude_team_id')
        
        # Get all channels from Slack
        all_channels = get_available_slack_channels()
        
        # If a team_id is provided, we want to include its currently linked channel
        # even if it's technically "linked" (so user can see their current selection)
        if exclude_team_id:
            current_link = project_slack_channels_collection.find_one({"team_id": exclude_team_id})
            if current_link:
                current_channel_id = current_link.get("channel_id")
                
                # Check if current channel is in the list, if not, fetch and add it
                if not any(ch["id"] == current_channel_id for ch in all_channels):
                    channel_info = get_slack_channel_info(current_channel_id)
                    if channel_info:
                        all_channels.append({
                            "id": current_channel_id,
                            "name": channel_info["channel_name"],
                            "is_private": channel_info["is_private"],
                            "is_member": True,
                            "currently_linked": True  # Flag to indicate this is the current selection
                        })
        
        return jsonify({
            "success": True,
            "channels": all_channels,
            "count": len(all_channels)
        })
        
    except Exception as e:
        print(f"❌ Error fetching available channels: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


@app.route('/api/team-analytics/<team_id>', methods=['GET'])
def get_team_analytics(team_id):
    """
    Get real-time analytics for a specific team
    """
    try:
        analytics = team_analytics_collection.find_one(
            {"team_id": team_id, "period": "last_24h"},
            {"_id": 0}
        )
        
        if analytics:
            return jsonify({"success": True, "data": analytics})
        else:
            return jsonify({"success": False, "message": "No analytics found for this team"})
            
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/api/team-messages/<team_id>', methods=['GET'])
def get_team_messages(team_id):
    """
    Get recent messages for a specific team
    """
    try:
        limit = int(request.args.get('limit', 50))
        
        messages = list(slack_messages_collection.find(
            {"team_id": team_id},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        
        return jsonify({"success": True, "data": messages, "count": len(messages)})
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/api/link-channel', methods=['POST'])
def link_channel_to_project():
    """
    Link a Slack channel to a project/team
    Request body: {"team_id": "...", "channel_id": "C09ABC...", "project_title": "...", "lead_username": "..."}
    """
    try:
        data = request.json
        team_id = data.get("team_id")
        channel_id = data.get("channel_id")
        project_title = data.get("project_title")
        lead_username = data.get("lead_username")
        
        if not all([team_id, channel_id, project_title, lead_username]):
            return jsonify({
                "success": False,
                "message": "team_id, channel_id, project_title, and lead_username are required"
            })
        
        # Check if channel is already linked to a DIFFERENT team
        existing_link = project_slack_channels_collection.find_one({"channel_id": channel_id})
        if existing_link and existing_link.get("team_id") != team_id:
            return jsonify({
                "success": False,
                "message": f"This channel is already linked to another team: {existing_link.get('project_title', 'Unknown')}"
            })
        
        # Remove any previous channel link for this team
        project_slack_channels_collection.delete_many({"team_id": team_id})
        
        # Create new link in ProjectSlackChannels collection
        link_doc = {
            "team_id": team_id,
            "channel_id": channel_id,
            "project_id": project_title,
            "project_title": project_title,
            "lead_username": lead_username,
            "linked_at": time.time(),
            "linked_date": datetime.now().isoformat()
        }
        
        project_slack_channels_collection.insert_one(link_doc)
        
        # Also update the UserTeams collection
        user_teams_collection.update_one(
            {"id": team_id},
            {"$set": {"slack_channel_id": channel_id, "linked_at": time.time()}}
        )
        
        return jsonify({
            "success": True,
            "message": "Channel linked successfully",
            "channel_id": channel_id
        })
            
    except Exception as e:
        print(f"❌ Error linking channel: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


@app.route('/api/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    """
    Get overall statistics for all teams
    """
    try:
        # Count total messages in last 24h
        cutoff = time.time() - 86400
        total_messages = slack_messages_collection.count_documents({"created_at": {"$gte": cutoff}})
        
        # Count active teams
        active_teams = user_teams_collection.count_documents({"status": "active"})
        
        # Get all analytics
        all_analytics = list(team_analytics_collection.find(
            {"period": "last_24h"},
            {"_id": 0}
        ))
        
        stats = {
            "total_messages_24h": total_messages,
            "active_teams": active_teams,
            "total_analytics_records": len(all_analytics),
            "teams": all_analytics
        }
        
        return jsonify({"success": True, "data": stats})
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Slack Listener",
        "timestamp": time.time()
    })


# ============================================================================
# STARTUP & MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 PHASE 2: SLACK INTEGRATION SYSTEM")
    print("="*70)
    print("\n📋 System Status:")
    print(f"   ✓ MongoDB Connected: {db.name}")
    print(f"   ✓ Collections Ready:")
    print(f"      - SlackUsers: User cache")
    print(f"      - SlackMessages: Message storage")
    print(f"      - ProjectSlackChannels: Channel-Project links")
    print(f"      - TeamAnalytics: Real-time metrics")
    print(f"\n⚙️  Starting background analytics updater...")
    
    # Start background analytics thread
    analytics_thread = threading.Thread(target=scheduled_analytics_update, daemon=True)
    analytics_thread.start()
    
    print(f"   ✓ Analytics updater running (5-minute intervals)")
    print(f"\n🌐 Starting Flask server on http://0.0.0.0:5001")
    print(f"   Webhook endpoint: /webhooks/slack")
    print(f"   Make sure to configure this URL in Slack Event Subscriptions")
    print(f"   Example: https://YOUR-NGROK-URL/webhooks/slack")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
