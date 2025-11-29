"""Observability helpers for logging, tracing, and metrics."""

from __future__ import annotations

import contextvars
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from functools import wraps
from typing import Any, Optional, Dict, Union, Any

from mcp.server.fastmcp import FastMCP
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pythonjsonlogger.json import JsonFormatter

from .models import TokenUsage

TRACE_ID = contextvars.ContextVar("mdapflow_trace_id", default=None)
MDAP_EXECUTION_ID = contextvars.ContextVar("mdapflow_execution_id", default="")
LOGGER = logging.getLogger("ensample")
TRACER = None


def configure_logging() -> logging.Logger:
    """Configure application-wide structured logging."""
    level = os.environ.get("MDAP_LOG_LEVEL", "INFO").upper()
    LOGGER.setLevel(level)
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        formatter = JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    return LOGGER


def configure_tracing() -> None:
    """Configure OpenTelemetry tracing when OTEL endpoint is supplied."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "mdapflow-mcp"}))
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    global TRACER
    TRACER = trace.get_tracer("ensample")


def _with_trace(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Add trace context to logging data."""
    data: dict[str, Any] = {}
    if extra:
        data.update(extra)
    
    trace_id = TRACE_ID.get()
    if trace_id:
        data["trace_id"] = trace_id
    
    execution_id = MDAP_EXECUTION_ID.get()
    if execution_id:
        data["execution_id"] = execution_id
    
    return data


def set_execution_context(execution_id: str) -> None:
    """Set the MDAP execution context for tracing."""
    MDAP_EXECUTION_ID.set(execution_id)


def clear_execution_context() -> None:
    """Clear the MDAP execution context."""
    try:
        MDAP_EXECUTION_ID.set("")
    except:
        pass


def instrumented_tool(server: FastMCP, *tool_args, **tool_kwargs):
    """Wrap FastMCP tools with structured logging and per-call trace IDs."""

    tool_decorator = server.tool(*tool_args, **tool_kwargs)

    def decorator(func: Callable[..., Awaitable[Any]]):
        tool_name = tool_kwargs.get("name", func.__name__)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())
            token = TRACE_ID.set(trace_id)
            execution_id = str(uuid.uuid4())
            exec_token = MDAP_EXECUTION_ID.set(execution_id)
            
            start = time.perf_counter()
            LOGGER.info("tool.start", extra=_with_trace({"tool": tool_name}))
            
            try:
                span_cm = (
                    TRACER.start_as_current_span(f"tool.{tool_name}") if TRACER else nullcontext()
                )
                with span_cm as span:
                    result = await func(*args, **kwargs)
                    duration = time.perf_counter() - start
                    
                    # Log tool success with enhanced context
                    LOGGER.info(
                        "tool.success",
                        extra=_with_trace({
                            "tool": tool_name, 
                            "duration": duration,
                            "execution_id": execution_id
                        }),
                    )
                    return result
            except Exception as e:
                duration = time.perf_counter() - start
                LOGGER.exception(
                    "tool.error",
                    extra=_with_trace({
                        "tool": tool_name, 
                        "duration": duration,
                        "execution_id": execution_id,
                        "error_type": type(e).__name__
                    }),
                )
                raise
            finally:
                TRACE_ID.reset(token)
                MDAP_EXECUTION_ID.reset(exec_token)

        return tool_decorator(wrapper)

    return decorator


# Token and Cost Tracking Helpers

def log_llm_call_start(
    provider: str, 
    model: str, 
    prompt_tokens: Optional[int] = None,
    prompt_length: Optional[int] = None,
    execution_id: Optional[str] = None
) -> None:
    """Log the start of an LLM API call."""
    extra_data: dict[str, Any] = {
        "event": "llm_call_start",
        "provider": provider,
        "model": model,
    }
    
    if prompt_tokens:
        extra_data["prompt_tokens"] = prompt_tokens
    if prompt_length:
        extra_data["prompt_length"] = prompt_length
    if execution_id:
        extra_data["execution_id"] = execution_id
        
    LOGGER.info("llm.call.start", extra=_with_trace(extra_data))


def log_llm_call_success(
    provider: str,
    model: str,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    latency_ms: Optional[float] = None,
    pricing_source: Optional[str] = None,
    execution_id: Optional[str] = None
) -> None:
    """Log successful LLM API call with token and cost data."""
    extra_data: dict[str, Any] = {
        "event": "llm_call_success",
        "provider": provider,
        "model": model,
    }
    
    if completion_tokens:
        extra_data["completion_tokens"] = completion_tokens
    if total_tokens:
        extra_data["total_tokens"] = total_tokens
    if cost_usd is not None:
        extra_data["cost_usd"] = cost_usd
    if latency_ms:
        extra_data["latency_ms"] = latency_ms
    if pricing_source:
        extra_data["pricing_source"] = pricing_source
    if execution_id:
        extra_data["execution_id"] = execution_id
        
    LOGGER.info("llm.call.success", extra=_with_trace(extra_data))


def log_llm_call_failure(
    provider: str,
    model: str,
    error_message: str,
    error_type: str,
    latency_ms: Optional[float] = None,
    execution_id: Optional[str] = None
) -> None:
    """Log failed LLM API call."""
    extra_data: dict[str, Any] = {
        "event": "llm_call_failure",
        "provider": provider,
        "model": model,
        "error_message": error_message,
        "error_type": error_type,
    }
    
    if latency_ms:
        extra_data["latency_ms"] = latency_ms
    if execution_id:
        extra_data["execution_id"] = execution_id
        
    LOGGER.error("llm.call.failure", extra=_with_trace(extra_data))


def log_mdap_execution_summary(
    role_name: str,
    voting_k: int,
    voting_rounds: int,
    total_llm_calls: int,
    total_tokens: int,
    total_cost_usd: float,
    success: bool = True,
    execution_id: Optional[str] = None
) -> None:
    """Log summary of MDAP execution with token and cost totals."""
    extra_data: dict[str, Any] = {
        "event": "mdap_execution_summary",
        "role_name": role_name,
        "voting_k": voting_k,
        "voting_rounds": voting_rounds,
        "total_llm_calls": total_llm_calls,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost_usd,
        "success": success,
    }
    
    if execution_id:
        extra_data["execution_id"] = execution_id
        
    if success:
        LOGGER.info("mdap.execution.success", extra=_with_trace(extra_data))
    else:
        LOGGER.error("mdap.execution.failure", extra=_with_trace(extra_data))


def add_token_span_attributes(
    span: Any, 
    provider: str,
    model: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    pricing_source: Optional[str] = None,
    prompt_length: Optional[int] = None
) -> None:
    """Add token usage and cost attributes to OpenTelemetry span."""
    if not span:
        return
        
    # Provider and model info
    span.set_attribute("llm.provider", provider)
    span.set_attribute("llm.model", model)
    
    # Token usage
    if prompt_tokens is not None:
        span.set_attribute("llm.tokens.prompt", prompt_tokens)
    if completion_tokens is not None:
        span.set_attribute("llm.tokens.completion", completion_tokens)
    if total_tokens is not None:
        span.set_attribute("llm.tokens.total", total_tokens)
    
    # Cost information
    if cost_usd is not None:
        span.set_attribute("llm.cost.usd", cost_usd)
    if pricing_source:
        span.set_attribute("llm.cost.source", pricing_source)
    
    # Prompt information
    if prompt_length is not None:
        span.set_attribute("llm.prompt.length", prompt_length)


def create_llm_span(
    operation: str,
    provider: str,
    model: str,
    prompt_tokens: Optional[int] = None,
    prompt_length: Optional[int] = None,
    execution_id: Optional[str] = None
) -> tuple[Any, dict[str, Any]]:
    """Create an LLM span with proper attributes and return span context."""
    if not TRACER:
        return nullcontext(), {}
    
    # Create span attributes
    attributes: dict[str, Any] = {
        "llm.provider": provider,
        "llm.model": model,
        "operation": operation,
    }
    
    if prompt_tokens is not None:
        attributes["llm.tokens.prompt"] = prompt_tokens
    if prompt_length is not None:
        attributes["llm.prompt.length"] = prompt_length
    if execution_id:
        attributes["execution_id"] = execution_id
    
    span = TRACER.start_span(f"llm.{operation}", attributes=attributes)
    return span, attributes


def log_red_flag_hit(
    rule_type: str,
    rule_value: Optional[str] = None,
    context: Optional[str] = None,
    execution_id: Optional[str] = None
) -> None:
    """Log when a red flag rule is hit."""
    extra_data: dict[str, Any] = {
        "event": "red_flag_hit",
        "rule_type": rule_type,
    }
    
    if rule_value:
        extra_data["rule_value"] = rule_value
    if context:
        extra_data["rule_context"] = context
    if execution_id:
        extra_data["execution_id"] = execution_id
        
    LOGGER.warning("red.flag.hit", extra=_with_trace(extra_data))


def log_voting_round(
    round_number: int,
    valid_responses: int,
    total_responses: int,
    consensus_reached: bool = False,
    execution_id: Optional[str] = None
) -> None:
    """Log voting round results."""
    extra_data: dict[str, Any] = {
        "event": "voting_round",
        "round_number": round_number,
        "valid_responses": valid_responses,
        "total_responses": total_responses,
        "consensus_reached": consensus_reached,
    }
    
    if execution_id:
        extra_data["execution_id"] = execution_id
        
    LOGGER.info("voting.round", extra=_with_trace(extra_data))


__all__ = [
    "LOGGER", 
    "configure_logging", 
    "configure_tracing", 
    "instrumented_tool", 
    "_with_trace",
    "set_execution_context",
    "clear_execution_context",
    "log_llm_call_start",
    "log_llm_call_success", 
    "log_llm_call_failure",
    "log_mdap_execution_summary",
    "add_token_span_attributes",
    "create_llm_span",
    "log_red_flag_hit",
    "log_voting_round"
]
