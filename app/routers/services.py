"""Services tab endpoints — supervisord process health for infra + lloyd services."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.supervisor_client import (
    _INFRA_SERVICES,
    _LLOYD_SERVICES,
    _supervisor_all,
    _supervisor_proxy,
    _port_open,
    _sup_state,
    _health,
    _read_log_tail,
)


router = APIRouter()


@router.get("/api/services")
async def get_services():
    procs = _supervisor_all()
    now = datetime.now().isoformat()
    services = []
    for sid, (name, port) in _INFRA_SERVICES.items():
        proc = procs.get(sid)
        active, sub = _sup_state(proc)
        port_healthy = _port_open(port) if port else None
        services.append({
            "id": sid,
            "name": name,
            "unit": sid,
            "port": port or 0,
            "systemdState": active,
            "portHealthy": bool(port_healthy) if port_healthy is not None else False,
            "health": _health(active, port_healthy),
        })
    return JSONResponse({"services": services, "timestamp": now})


@router.get("/api/services/detail")
async def get_service_detail(id: str = ""):
    if not id or id not in _INFRA_SERVICES:
        raise HTTPException(status_code=404, detail=f"Service not found: {id}")
    name, port = _INFRA_SERVICES[id]
    procs = _supervisor_all()
    proc = procs.get(id, {})
    active, sub = _sup_state(proc)
    pid = proc.get("pid") or None
    log_path = f"/home/alansrobotlab/agent-services/logs/{id}.log"
    log_lines = _read_log_tail(log_path)
    raw = f"state={proc.get('statename','?')} pid={pid} desc={proc.get('description','')}"
    return JSONResponse({
        "id": id,
        "name": name,
        "unit": id,
        "port": port or 0,
        "pid": pid,
        "memory": None,
        "cpu": None,
        "tasks": None,
        "activeSince": proc.get("description", None),
        "logLines": log_lines,
        "rawStatus": raw,
    })


@router.post("/api/services/action")
async def service_action(request: Request):
    data = await request.json()
    service_id = data.get("serviceId", "")
    action = data.get("action", "")
    all_services = {**_INFRA_SERVICES, **_LLOYD_SERVICES}
    if service_id not in all_services:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service_id}")
    if action not in ("start", "stop", "restart"):
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
    try:
        proxy = _supervisor_proxy()
        if action == "start":
            proxy.supervisor.startProcess(service_id)
        elif action == "stop":
            proxy.supervisor.stopProcess(service_id)
        elif action == "restart":
            try:
                proxy.supervisor.stopProcess(service_id)
            except Exception:
                pass
            proxy.supervisor.startProcess(service_id)
        return JSONResponse({"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/agent-services")
async def get_agent_services():
    procs = _supervisor_all()
    now = datetime.now().isoformat()
    services = []
    for sid, (name, port) in _LLOYD_SERVICES.items():
        proc = procs.get(sid)
        active, sub = _sup_state(proc)
        port_healthy = _port_open(port) if port else None
        if port_healthy:
            active, sub = "active", "running"
        uptime = proc.get("description") if proc else None
        services.append({
            "id": sid,
            "unit": sid,
            "name": name,
            "activeState": active,
            "subState": sub,
            "port": port,
            "portHealthy": port_healthy,
            "uptime": uptime,
            "health": _health(active, port_healthy),
        })
    return JSONResponse({"services": services, "timestamp": now})


@router.get("/api/agent-services/detail")
async def get_agent_service_detail(unit: str = ""):
    if not unit or unit not in _LLOYD_SERVICES:
        raise HTTPException(status_code=404, detail=f"Service not found: {unit}")
    name, port = _LLOYD_SERVICES[unit]
    procs = _supervisor_all()
    proc = procs.get(unit, {})
    pid = proc.get("pid") or None
    log_path = f"/home/alansrobotlab/lloyd/logs/{unit.replace('lloyd-', '')}.log"
    log_lines = _read_log_tail(log_path)
    raw = f"state={proc.get('statename','?')} pid={pid} desc={proc.get('description','')}"
    return JSONResponse({
        "unit": unit,
        "name": name,
        "pid": pid,
        "memory": None,
        "cpu": None,
        "tasks": None,
        "activeSince": proc.get("description", None),
        "logLines": log_lines,
        "rawStatus": raw,
    })
