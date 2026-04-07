import os
import json
import logging
import asyncio
import re
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(title="LLM Integrate SNS", version="1.0.0")

# Environment variables
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PORT = int(os.getenv("PORT", 8080))

# AWS Clients
try:
    cloudwatch_logs_client = boto3.client(
        'logs',
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    sns_client = boto3.client(
        'sns',
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
except Exception as e:
    logger.error(f"Failed to initialize AWS clients: {e}")
    cloudwatch_logs_client = None
    sns_client = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "LLM Integrate SNS service is running"}


async def handle_subscription_confirmation(message: Dict[str, Any]) -> Dict[str, str]:
    """
    Handle SNS SubscriptionConfirmation message.
    Automatically confirm subscription by making HTTP GET to subscribe URL.
    """
    try:
        subscribe_url = message.get("SubscribeURL")
        if not subscribe_url:
            logger.error("SubscribeURL not found in message")
            return {"status": "error", "message": "SubscribeURL not found"}

        # Make HTTP GET request to confirm subscription
        response = requests.get(subscribe_url, timeout=30)
        
        if response.status_code == 200:
            logger.info(f"Successfully confirmed SNS subscription")
            return {"status": "success", "message": "Subscription confirmed"}
        else:
            logger.error(f"Failed to confirm subscription: HTTP {response.status_code}")
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    
    except Exception as e:
        logger.error(f"Error confirming subscription: {str(e)}")
        return {"status": "error", "message": str(e)}


async def get_cloudwatch_error_logs(log_group: str, log_stream: Optional[str] = None) -> list:
    """
    Fetch error logs from CloudWatch Logs.
    Filters for "ERROR" keyword within the last 1 hour, returns 5 most recent logs.
    """
    try:
        if not cloudwatch_logs_client:
            logger.error("CloudWatch Logs client not initialized")
            return []

        # Calculate time range: last 3 hours to avoid missing delayed alarms.
        end_time = int(datetime.utcnow().timestamp() * 1000)
        start_time = int((datetime.utcnow() - timedelta(hours=3)).timestamp() * 1000)

        query = (
            "fields @timestamp, @message "
            "| filter @message like /ERROR|Exception|Traceback|Task timed out|Runtime\\.[A-Za-z_]+|ValueError|ValidationError|error_message|Missing or invalid required field|Bad POST request/ "
            "| sort @timestamp desc "
            "| limit 20"
        )

        # Start query execution
        response = cloudwatch_logs_client.start_query(
            logGroupName=log_group,
            startTime=int(start_time / 1000),
            endTime=int(end_time / 1000),
            queryString=query
        )

        query_id = response["queryId"]
        
        # Poll for query completion
        max_attempts = 30
        attempts = 0
        while attempts < max_attempts:
            result = cloudwatch_logs_client.get_query_results(queryId=query_id)
            status = result["status"]
            
            if status == "Complete":
                logs = []
                for record in result["results"]:
                    raw_log_entry = {
                        field["field"]: field["value"]
                        for field in record
                    }
                    logs.append(normalize_log_entry(raw_log_entry))

                if logs:
                    logger.info(f"Retrieved {len(logs)} error logs from {log_group}")
                    return logs

                # Fallback: if Insights returns empty, pull recent events and extract error-like lines.
                fallback_logs = cloudwatch_logs_client.filter_log_events(
                    logGroupName=log_group,
                    startTime=start_time,
                    endTime=end_time,
                    filterPattern='?ERROR ?Exception ?Traceback ?"Task timed out" ?"Runtime." ?ValueError ?ValidationError ?error_message ?"Missing or invalid required field" ?"Bad POST request"',
                    limit=50
                )

                events = fallback_logs.get("events", [])
                logs = [
                    normalize_log_entry(
                        {
                            "timestamp": str(event.get("timestamp", "N/A")),
                            "message": event.get("message", "")
                        }
                    )
                    for event in events
                ]
                
                logger.info(f"Retrieved {len(logs)} error logs from {log_group}")
                return logs
            elif status in ["Failed", "Cancelled"]:
                logger.error(f"CloudWatch Logs query failed with status: {status}")
                return []
            
            attempts += 1
            await asyncio.sleep(0.5)
        
        logger.warning("CloudWatch Logs query timeout")
        return []

    except Exception as e:
        logger.error(f"Error fetching CloudWatch logs from {log_group}: {str(e)}")
        return []


async def get_llm_analysis(logs: list) -> Dict[str, str]:
    """
    Call LLM (ollama or groq) to analyze error logs and provide:
    1. Summary: cause of the error
    2. Solution: recommendation to fix
    """
    try:
        # Format logs for LLM prompt
        if logs:
            logs_text = build_llm_logs_text(logs)
        else:
            logs_text = (
                "No CloudWatch error logs were retrieved for this alarm in the selected "
                "time window. Provide likely causes and troubleshooting steps for this condition."
            )

        prompt = f"""Sebagai DevOps, berikan 1 ringkasan penyebab error (Summary) dan 1 rekomendasi (Solusi) dari semua log berikut:

ERROR LOGS:
{logs_text}

Berikan response dalam format JSON dengan keys "summary" dan "solution"."""

        if LLM_PROVIDER == "ollama":
            result = await call_ollama(prompt)
        elif LLM_PROVIDER == "groq":
            result = await call_groq(prompt)
        else:
            logger.error(f"Unknown LLM provider: {LLM_PROVIDER}")
            return {"summary": "Unknown LLM provider", "solution": "Please check configuration"}

        return result

    except Exception as e:
        logger.error(f"Error getting LLM analysis: {str(e)}")
        return {"summary": "Error during analysis", "solution": str(e)}


def extract_error_detail(message: str) -> str:
    """
    Extract core error detail from CloudWatch message.
    Supports JSON payload in message and Python traceback patterns.
    """
    if not message:
        return ""

    cleaned = message.replace("\xa0", " ").strip()

    # Some services wrap the actual payload under a text prefix.
    payload_candidate = cleaned
    if "\t" in cleaned:
        payload_candidate = cleaned.split("\t")[-1].strip()

    # Prefer structured JSON error payload when available.
    json_start = payload_candidate.find("{")
    json_end = payload_candidate.rfind("}")
    if json_start != -1 and json_end > json_start:
        json_candidate = payload_candidate[json_start:json_end + 1]
        try:
            parsed = json.loads(json_candidate)
            if isinstance(parsed.get("body"), str):
                try:
                    parsed_body = json.loads(parsed["body"])
                    if isinstance(parsed_body, dict):
                        parsed = parsed_body
                except json.JSONDecodeError:
                    pass

            error_message = (
                parsed.get("error_message")
                or parsed.get("errorMessage")
                or parsed.get("message")
            )
            error_type = (
                parsed.get("error_type")
                or parsed.get("errorType")
                or parsed.get("type")
            )
            if error_type and error_message:
                return f"{error_type}: {error_message}"
            if error_message:
                return error_message
        except json.JSONDecodeError:
            pass

    traceback_match = re.search(r"([A-Za-z_]+Error):\s*(.+)", cleaned)
    if traceback_match:
        return f"{traceback_match.group(1)}: {traceback_match.group(2).strip()}"

    return cleaned


def normalize_log_entry(log_entry: Dict[str, str]) -> Dict[str, str]:
    """Normalize CloudWatch result fields to stable keys across query variants."""
    timestamp = (
        log_entry.get("timestamp")
        or log_entry.get("@timestamp")
        or "N/A"
    )
    message = (
        log_entry.get("message")
        or log_entry.get("@message")
        or ""
    )

    normalized = {
        "timestamp": timestamp,
        "message": message,
        "error": extract_error_detail(message)
    }
    return normalized


def build_llm_logs_text(logs: list) -> str:
    """Build compact, deduplicated error context for LLM prompt."""
    seen = set()
    lines = []

    for log in logs:
        timestamp = log.get("timestamp") or log.get("@timestamp") or "N/A"
        message = log.get("error") or log.get("message") or log.get("@message") or ""
        compact = " ".join(message.split())
        if not compact or compact in seen:
            continue
        seen.add(compact)
        lines.append(f"[{timestamp}] {compact}")

    if not lines:
        return "No parsable error logs found"

    return "\n".join(lines[:10])


async def call_ollama(prompt: str) -> Dict[str, str]:
    """Call Ollama API for text generation"""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=60)
        
        if response.status_code != 200:
            logger.error(f"Ollama API error: HTTP {response.status_code}")
            return {"summary": "Ollama error", "solution": "Failed to get response"}
        
        result = response.json()
        response_text = result.get("response", "")
        
        # Parse JSON response
        try:
            parsed = json.loads(response_text)
            return {
                "summary": parsed.get("summary", response_text),
                "solution": parsed.get("solution", "")
            }
        except json.JSONDecodeError:
            # If not JSON, return as summary
            return {"summary": response_text, "solution": ""}

    except Exception as e:
        logger.error(f"Error calling Ollama: {str(e)}")
        return {"summary": "Ollama error", "solution": str(e)}


async def call_groq(prompt: str) -> Dict[str, str]:
    """Call Groq API for text generation"""
    try:
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY not configured")
            return {"summary": "Groq API key missing", "solution": "Configuration error"}

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            logger.error(f"Groq API error: HTTP {response.status_code} - {response.text}")
            return {"summary": "Groq error", "solution": f"HTTP {response.status_code}"}
        
        result = response.json()
        response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Parse JSON response
        try:
            parsed = json.loads(response_text)
            return {
                "summary": parsed.get("summary", response_text),
                "solution": parsed.get("solution", "")
            }
        except json.JSONDecodeError:
            # If not JSON, return as summary
            return {"summary": response_text, "solution": ""}

    except Exception as e:
        logger.error(f"Error calling Groq: {str(e)}")
        return {"summary": "Groq error", "solution": str(e)}


async def publish_to_sns(alarm_name: str, summary: str, solution: str) -> bool:
    """
    Publish analysis result to SNS Topic with subject about the alarm.
    """
    try:
        if not sns_client or not SNS_TOPIC_ARN:
            logger.error("SNS client or topic ARN not configured")
            return False

        message_body = f"""
INCIDENT REPORT
===============

Alarm: {alarm_name}
Time: {datetime.utcnow().isoformat()}

SUMMARY (Penyebab Error):
{summary}

SOLUTION (Rekomendasi):
{solution}
"""

        subject = f"Resume Incident Report: {alarm_name}"

        response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message_body
        )

        logger.info(f"Published to SNS: MessageId={response.get('MessageId')}")
        return True

    except Exception as e:
        logger.error(f"Error publishing to SNS: {str(e)}")
        return False


def extract_log_group_from_alarm(alarm_name: str) -> Optional[str]:
    """
    Extract log group from alarm name using LIST_SNS_TOPIC_ARN mapping.
    """
    try:
        list_config = os.getenv("LIST_SNS_TOPIC_ARN", "{}")
        mapping = json.loads(list_config)
        
        for alarm_key, log_group in mapping.items():
            if alarm_key.lower() in alarm_name.lower():
                return log_group
        
        return None
    
    except Exception as e:
        logger.error(f"Error extracting log group from alarm: {str(e)}")
        return None


async def handle_notification(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle SNS Notification message.
    1. Extract AlarmName
    2. Get log group from mapping
    3. Fetch error logs from CloudWatch
    4. Call LLM for analysis
    5. Publish result to SNS
    """
    try:
        # Parse the Sns message if it's a string
        sns_message = message.get("Message", "{}")
        if isinstance(sns_message, str):
            try:
                sns_message = json.loads(sns_message)
            except json.JSONDecodeError:
                sns_message = {"raw": sns_message}
        
        # Extract alarm name from different possible fields
        alarm_name = (
            sns_message.get("AlarmName") or
            message.get("Subject", "Unknown Alarm")
        )
        
        logger.info(f"Processing notification for alarm: {alarm_name}")

        # Get log group from mapping
        log_group = extract_log_group_from_alarm(alarm_name)
        
        if not log_group:
            logger.warning(f"No log group mapping found for alarm: {alarm_name}")
            log_group = f"/aws/lambda/{alarm_name}"

        logger.info(f"Using log group: {log_group}")

        # Fetch error logs from CloudWatch
        error_logs = await get_cloudwatch_error_logs(log_group)
        
        if not error_logs:
            logger.warning(f"No error logs found in {log_group}")

        # Get LLM analysis
        analysis = await get_llm_analysis(error_logs)
        
        logger.info(f"LLM Analysis complete - Summary: {analysis['summary'][:100]}...")

        # Publish to SNS
        publish_result = await publish_to_sns(
            alarm_name,
            analysis["summary"],
            analysis["solution"]
        )

        return {
            "status": "success" if publish_result else "partial",
            "alarm_name": alarm_name,
            "log_group": log_group,
            "logs_count": len(error_logs),
            "analysis": analysis
        }

    except Exception as e:
        logger.error(f"Error handling notification: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/webhook")
async def webhook(request: Request):
    """
    Main webhook endpoint to handle SNS notifications.
    Supports:
    - SubscriptionConfirmation: Auto-confirms subscription
    - Notification: Processes alarm and sends analysis to LLM
    """
    try:
        body = await request.json()
        logger.info(f"Received webhook: {body.get('Type')}")

        message_type = body.get("Type")

        if message_type == "SubscriptionConfirmation":
            result = await handle_subscription_confirmation(body)
            return JSONResponse(result, status_code=200)

        elif message_type == "Notification":
            result = await handle_notification(body)
            status_code = 200 if result.get("status") == "success" else 202
            return JSONResponse(result, status_code=status_code)

        else:
            logger.warning(f"Unknown message type: {message_type}")
            return JSONResponse(
                {"status": "error", "message": f"Unknown message type: {message_type}"},
                status_code=400
            )

    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    
    # Import asyncio for the async operations
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)
