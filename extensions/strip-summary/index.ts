export default function register(api: any) {
  api.logger?.info?.("strip-summary: plugin registered");

  api.on("message_sending", (event: any, ctx: any) => {
    api.logger?.info?.(`strip-summary: hook fired — channel=${ctx.channelId}, contentLen=${event.content?.length}`);

    // skip webchat — voice pipeline needs summary blocks
    if (ctx.channelId === "webchat") {
      api.logger?.info?.("strip-summary: skipping webchat");
      return;
    }

    const stripped = event.content
      .replace(/<summary>[\s\S]*?<\/summary>\s*/g, "")
      .trim();

    api.logger?.info?.(`strip-summary: stripped ${event.content.length - stripped.length} chars`);

    if (!stripped) {
      return { cancel: true };
    }

    return { content: stripped };
  });
}
