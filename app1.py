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


def fetch_and_store_channel_history(channel_id, team_id, lead_username, project_title, oldest=None):
    """
    Fetch message history directly from Slack API for a channel and store in MongoDB.
    Used as a fallback when webhook isn't delivering messages.
    Returns: number of new messages stored
    """
    url = "https://slack.com/api/conversations.history"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    
    if oldest is None:
        oldest = str(time.time() - 86400)  # Last 24 hours
    
    params = {
        "channel": channel_id,
        "oldest": oldest,
        "limit": 200
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        
        if not data.get("ok"):
            error = data.get("error", "unknown")
            print(f"❌ Failed to fetch history for channel {channel_id}: {error}")
            if error == "not_in_channel":
                print(f"   ⚠️  Bot is NOT a member of channel {channel_id}. Invite the bot to this channel!")
            return 0
        
        messages = data.get("messages", [])
        new_count = 0
        
        for msg in messages:
            # Skip bot messages, subtypes (joins, leaves, etc.)
            if msg.get("subtype") or not msg.get("user"):
                continue
            
            ts = msg.get("ts", "")
            
            # Check if already stored (avoid duplicates)
            existing = slack_messages_collection.find_one({
                "channel_id": channel_id,
                "timestamp": float(ts)
            })
            if existing:
                continue
            
            user_id = msg["user"]
            user_name = resolve_slack_user(user_id)
            text = msg.get("text", "")
            
            message_doc = {
                "slack_user_id": user_id,
                "user_name": user_name,
                "channel_id": channel_id,
                "project_title": project_title,
                "project_id": project_title,
                "team_id": team_id,
                "lead_username": lead_username,
                "message": text,
                "text": text,
                "timestamp": float(ts),
                "datetime": datetime.fromtimestamp(float(ts)).isoformat(),
                "created_at": float(ts),  # Use message timestamp so 24h filter works correctly
                "date": datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d"),
                "time": datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S"),
                "source": "history_fetch"
            }
            
            slack_messages_collection.insert_one(message_doc)
            new_count += 1
        
        if new_count > 0:
            print(f"📥 Fetched {new_count} new messages from channel {channel_id} via API")
        
        return new_count
        
    except Exception as e:
        print(f"❌ Error fetching channel history for {channel_id}: {str(e)}")
        return 0


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
        "created_at": float(timestamp),  # Use message timestamp (not ingest time) so 24h filter is accurate
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

def score_messages_behaviorally(messages_text: str) -> float:
    """
    Simple keyword-based behavioral scoring of concatenated Slack messages.
    Returns a score 0-100. Used when the RoBERTa model is unavailable.
    Positive signals increase score, negative signals decrease it.
    """
    if not messages_text or not messages_text.strip():
        return None  # No data

    text = messages_text.lower()
    base = 70.0

    positive_keywords = [
        "thanks", "thank you", "great", "good job", "well done", "appreciate",
        "agree", "helpful", "sure", "happy to", "absolutely", "excellent",
        "nice work", "awesome", "perfect", "will do", "on it", "done", "finished",
        "completed", "delivered", "let me know", "sounds good", "yes", "correct",
        "good point", "i can help"
    ]
    negative_keywords = [
        "no", "won't", "can't", "refuse", "disagree", "wrong", "not my job",
        "whatever", "doubt", "problem", "issue", "fail", "failed", "mistake",
        "error", "terrible", "hate", "bad", "ugh", "annoying", "frustrated",
        "late", "delay", "missed", "not done", "incomplete"
    ]

    pos_hits = sum(1 for kw in positive_keywords if kw in text)
    neg_hits = sum(1 for kw in negative_keywords if kw in text)

    score = base + (pos_hits * 1.5) - (neg_hits * 2.0)
    return round(min(max(score, 0.0), 100.0), 1)


def update_team_analytics(team_id, team_owner, project_title):
    """
    Calculate and store real-time analytics for a team.
    Computes:
      - basic message stats (total, active users, top contributors, peak hours)
      - per-member behavioral scores based on their Slack messages vs their initial score
      - team-level compatibility trend based on member score changes
    """
    if not team_id:
        return

    # ── 1. Get ALL stored messages for this team (not just 24h) ──────────────
    all_messages = list(slack_messages_collection.find({"team_id": team_id}))

    # ── 2. Get messages from last 24 hours for activity stats ─────────────────
    cutoff_time = time.time() - 86400
    recent_messages = [m for m in all_messages if m.get("created_at", 0) >= cutoff_time]

    if not all_messages:
        print(f"⚠️  No messages found for team {team_id} in last 24h")
        return

    total_messages = len(recent_messages)

    # ── 3. Basic activity stats ────────────────────────────────────────────────
    user_message_counts = defaultdict(int)
    for msg in recent_messages:
        user_message_counts[msg.get("user_name", "Unknown")] += 1

    unique_users = len(user_message_counts)
    top_contributors = sorted(user_message_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    hour_counts = defaultdict(int)
    for msg in recent_messages:
        try:
            hour = datetime.fromtimestamp(float(msg.get("timestamp", 0))).hour
            hour_counts[hour] += 1
        except Exception:
            pass
    peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    engagement_score = round(total_messages / unique_users, 2) if unique_users > 0 else 0

    # ── 4. Load the saved team from UserTeams collection ──────────────────────
    team_doc = db["UserTeams"].find_one({"id": team_id})
    team_members = team_doc.get("members", []) if team_doc else []
    # initial_compatibility is stored when team is saved
    initial_compat = float(team_doc.get("compatibility_score", 0)) if team_doc else 0.0

    # ── 5. Per-member behavioral analysis ────────────────────────────────────
    # Map Slack display names → team member names (fuzzy: lowercase strip match)
    def normalize_name(n):
        return n.lower().strip() if n else ""

    # Build a lookup: normalized_name → member dict
    member_lookup = {}
    for m in team_members:
        member_lookup[normalize_name(m.get("name", ""))] = m

    # Group ALL messages by user_name
    messages_by_user = defaultdict(list)
    for msg in all_messages:
        messages_by_user[msg.get("user_name", "Unknown")].append(msg.get("text", ""))

    # Get Slack channel members from actual Slack API to avoid showing non-members
    channel_link = project_slack_channels_collection.find_one({"team_id": team_id})
    channel_members_slack = set()
    if channel_link and channel_link.get("channel_id"):
        try:
            url = "https://slack.com/api/conversations.members"
            headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
            resp = requests.get(url, headers=headers,
                                params={"channel": channel_link["channel_id"]}, timeout=5)
            resp_data = resp.json()
            if resp_data.get("ok"):
                for uid in resp_data.get("members", []):
                    uname = resolve_slack_user(uid)
                    channel_members_slack.add(normalize_name(uname))
        except Exception as e:
            print(f"⚠️  Could not fetch channel members: {e}")

    # Build member behavioral analysis — only for actual team members
    member_behavioral_analysis = []
    for member in team_members:
        member_name = member.get("name", "")
        norm_name = normalize_name(member_name)
        initial_score = float(member.get("behavioral_score", 70.0))

        # Find messages from this member — try exact then partial name match
        user_texts = messages_by_user.get(member_name, [])
        if not user_texts:
            # Try matching Slack display name to team member name (partial)
            for slack_name, texts in messages_by_user.items():
                if (normalize_name(slack_name) == norm_name or
                        norm_name in normalize_name(slack_name) or
                        normalize_name(slack_name) in norm_name):
                    user_texts = texts
                    break

        has_data = len(user_texts) >= 1
        messages_analyzed = len(user_texts)

        if has_data:
            combined_text = " ".join(user_texts[:20])  # cap at 20 messages for scoring
            current_score = score_messages_behaviorally(combined_text)
            if current_score is None:
                current_score = initial_score
                has_data = False
        else:
            current_score = initial_score

        fluctuation = round(current_score - initial_score, 1)
        fluctuation_pct = round((fluctuation / initial_score * 100), 1) if initial_score > 0 else 0.0

        if has_data and messages_analyzed >= 2:
            if fluctuation > 3:
                trend = "improving"
            elif fluctuation < -3:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        alert = has_data and fluctuation < -10

        member_behavioral_analysis.append({
            "user_name": member_name,
            "initial_behavioral_score": round(initial_score, 1),
            "current_behavioral_score": round(current_score, 1),
            "fluctuation": fluctuation,
            "fluctuation_pct": fluctuation_pct,
            "trend": trend,
            "alert": alert,
            "has_data": has_data,
            "messages_analyzed": messages_analyzed
        })

    # ── 6. Team compatibility trend ───────────────────────────────────────────
    members_with_data = [m for m in member_behavioral_analysis if m["has_data"]]
    if members_with_data:
        avg_fluctuation = sum(m["fluctuation"] for m in members_with_data) / len(members_with_data)
        # Approximate current compatibility by adjusting initial
        current_compat = round(max(0.0, min(100.0, initial_compat + avg_fluctuation * 0.5)), 1)
        compat_change = round(current_compat - initial_compat, 1)
        if compat_change > 2:
            compat_trend = "improving"
        elif compat_change < -2:
            compat_trend = "declining"
        else:
            compat_trend = "stable"
    else:
        current_compat = initial_compat
        compat_change = 0.0
        compat_trend = "stable"

    team_compatibility = {
        "initial_score": round(initial_compat, 1),
        "current_score": current_compat,
        "change": compat_change,
        "trend": compat_trend
    }

    # ── 7. Save analytics document ────────────────────────────────────────────
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
        "member_behavioral_analysis": member_behavioral_analysis,
        "team_compatibility": team_compatibility,
        "last_updated": time.time(),
        "last_updated_iso": datetime.now().isoformat()
    }

    team_analytics_collection.update_one(
        {"team_id": team_id, "period": "last_24h"},
        {"$set": analytics_doc},
        upsert=True
    )

    print(f"📊 Analytics updated for team {team_id}: {total_messages} msgs, "
          f"{unique_users} active, compat {initial_compat}→{current_compat}")


# ============================================================================
# BACKGROUND ANALYTICS UPDATER (RUNS EVERY 5 MINUTES)
# ============================================================================

def scheduled_analytics_update():
    """
    Background job to update analytics for all active teams.
    Runs every 5 minutes.
    Also fetches message history directly from Slack API as a fallback
    in case the webhook is not delivering events (e.g. ngrok expired, bot not in channel).
    """
    while True:
        try:
            print("\n🔄 Running scheduled analytics update...")
            
            active_teams = list(user_teams_collection.find({"status": "active"}))
            
            for team in active_teams:
                team_id = team.get("id")
                team_owner = team.get("username")
                project_title = team.get("project_title")
                
                if not (team_id and team_owner and project_title):
                    continue
                
                # --- Fallback: fetch messages directly from Slack API ---
                # Look up the linked Slack channel for this team
                channel_link = project_slack_channels_collection.find_one({"team_id": team_id})
                if channel_link:
                    channel_id = channel_link.get("channel_id")
                    if channel_id:
                        oldest = str(time.time() - 86400)  # last 24h
                        fetched = fetch_and_store_channel_history(
                            channel_id=channel_id,
                            team_id=team_id,
                            lead_username=team_owner,
                            project_title=project_title,
                            oldest=oldest
                        )
                        if fetched > 0:
                            print(f"   ✅ Pulled {fetched} messages from Slack API for team {team_id}")
                else:
                    print(f"   ⚠️  Team {team_id} has no linked Slack channel. Use /api/link-channel to set one.")
                # --------------------------------------------------------
                
                update_team_analytics(team_id, team_owner, project_title)
            
            print("✅ Scheduled analytics update complete")
            
        except Exception as e:
            print(f"❌ Error in scheduled analytics: {str(e)}")
            import traceback
            traceback.print_exc()
        
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


@app.route('/api/fetch-channel-history/<team_id>', methods=['POST'])
def fetch_history_for_team(team_id):
    """
    Manually trigger a Slack history fetch for a team's linked channel.
    Called by the "Sync Slack Messages" button in the monitor UI.
    """
    try:
        channel_link = project_slack_channels_collection.find_one({"team_id": team_id})
        if not channel_link:
            return jsonify({"success": False, "message": "No Slack channel linked to this team"})

        channel_id = channel_link.get("channel_id")
        lead_username = channel_link.get("lead_username", "")
        project_title = channel_link.get("project_title", "")

        # Fetch last 7 days of history so existing messages are captured
        oldest = str(time.time() - 7 * 86400)
        new_count = fetch_and_store_channel_history(
            channel_id=channel_id,
            team_id=team_id,
            lead_username=lead_username,
            project_title=project_title,
            oldest=oldest
        )

        # Re-run analytics immediately after fetching
        team_doc = db["UserTeams"].find_one({"id": team_id})
        if team_doc:
            update_team_analytics(team_id, team_doc.get("username", ""), team_doc.get("project_title", ""))

        return jsonify({
            "success": True,
            "message": f"Synced {new_count} new messages from #{channel_link.get('channel_name', channel_id)}",
            "new_messages": new_count
        })
    except Exception as e:
        print(f"❌ Error in fetch_history_for_team: {e}")
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


@app.route('/api/diagnostics', methods=['GET'])
def diagnostics():
    """
    Diagnostic endpoint: checks Slack bot token validity, channel memberships,
    and message counts per team. Useful for debugging webhook delivery issues.
    """
    try:
        results = {}
        
        # 1. Check bot token
        auth_resp = requests.get(
            "https://slack.com/api/auth.test",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            timeout=5
        ).json()
        results["bot_token_valid"] = auth_resp.get("ok", False)
        results["bot_name"] = auth_resp.get("user", "unknown")
        if not auth_resp.get("ok"):
            results["token_error"] = auth_resp.get("error")
        
        # 2. Check each linked channel
        cutoff = time.time() - 86400
        channel_checks = []
        for link in project_slack_channels_collection.find():
            channel_id = link.get("channel_id")
            team_id = link.get("team_id")
            
            # Check bot membership
            ch_resp = requests.get(
                "https://slack.com/api/conversations.info",
                headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                params={"channel": channel_id},
                timeout=5
            ).json()
            
            is_member = False
            channel_name = channel_id
            if ch_resp.get("ok"):
                ch = ch_resp.get("channel", {})
                is_member = ch.get("is_member", False)
                channel_name = ch.get("name", channel_id)
            
            # Count stored messages in last 24h
            msg_count = slack_messages_collection.count_documents({
                "team_id": team_id,
                "created_at": {"$gte": cutoff}
            })
            
            channel_checks.append({
                "team_id": team_id,
                "channel_id": channel_id,
                "channel_name": channel_name,
                "bot_is_member": is_member,
                "messages_last_24h_in_db": msg_count,
                "warning": None if is_member else "⚠️ Bot is NOT in this channel! Run: /invite @<bot_name> in the channel."
            })
        
        results["channels"] = channel_checks
        
        return jsonify({"success": True, "data": results})
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