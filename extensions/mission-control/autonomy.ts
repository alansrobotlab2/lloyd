import type { IncomingMessage, ServerResponse } from "node:http";
import type { PluginContext } from "./types.js";
import { jsonResponse, readBody } from "./helpers.js";
import { getTasks, getTask, writeTask, deleteTask, runTask, getConfig, setConfig, getAllConfig, getRuns } from "./autonomy-service.js";

function parseQuery(req: IncomingMessage): URLSearchParams {
  const url = new URL(req.url ?? "/", "http://localhost");
  return url.searchParams;
}

export function registerAutonomyRoutes(ctx: PluginContext) {
  const { api } = ctx;

  api.registerHttpRoute({
    path: "/api/mc/autonomy-tasks",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const tasks = await getTasks();
        jsonResponse(res, { tasks });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  api.registerHttpRoute({
    path: "/api/mc/autonomy-write-task",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const data = JSON.parse(await readBody(req));
        const result = await writeTask(data);
        jsonResponse(res, result);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  api.registerHttpRoute({
    path: "/api/mc/autonomy-get-task",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const id = parseInt(parseQuery(req).get("id") || "0");
        const task = await getTask(id);
        jsonResponse(res, { task });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  api.registerHttpRoute({
    path: "/api/mc/autonomy-run-task",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const data = JSON.parse(await readBody(req));
        const result = await runTask(data.id);
        jsonResponse(res, result);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  api.registerHttpRoute({
    path: "/api/mc/autonomy-delete-task",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const data = JSON.parse(await readBody(req));
        const result = await deleteTask(data.id);
        jsonResponse(res, result);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  api.registerHttpRoute({
    path: "/api/mc/autonomy-config",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const params = parseQuery(req);
        const key = params.get("key");
        const value = params.get("value");
        if (key && value !== null) {
          await setConfig(key, value);
          jsonResponse(res, { [key]: value });
        } else if (key) {
          const val = await getConfig(key);
          jsonResponse(res, { [key]: val });
        } else {
          const all = await getAllConfig();
          jsonResponse(res, all);
        }
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  api.registerHttpRoute({
    path: "/api/mc/autonomy-runs",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const params = parseQuery(req);
        const taskId = parseInt(params.get("task_id") || "0");
        const limit = parseInt(params.get("limit") || "20");
        const runs = await getRuns(taskId, limit);
        jsonResponse(res, { runs });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });
}
