/**
 * Memory agent — vault knowledge management specialist.
 * Model: Haiku (cost-effective for vault operations)
 *
 * Handles vault search, note creation, knowledge organization,
 * tag management, and multi-source synthesis.
 */

const MCP_TOOLS = [
  "mcp__openclaw-tools__mem_search",
  "mcp__openclaw-tools__mem_get",
  "mcp__openclaw-tools__mem_write",
  "mcp__openclaw-tools__tag_search",
  "mcp__openclaw-tools__tag_explore",
  "mcp__openclaw-tools__vault_overview",
];

export const memoryAgent = {
  description:
    "Vault knowledge management specialist. Use for vault search, note creation, knowledge organization, " +
    "tag management, multi-source synthesis, and vault maintenance. Runs on Haiku for cost efficiency.",
  prompt: `# Memory Agent

You manage the Obsidian vault — searching, reading, writing, organizing, and maintaining knowledge.

## Workflow
1. Understand the knowledge task from your instructions
2. Search the vault first (mem_search, tag_search) to find existing content
3. Read relevant documents (mem_get) to understand context
4. Create, update, or reorganize vault content as needed (mem_write)
5. Use tag_explore to discover connections between topics

## Principles
- Search before creating — avoid duplicates
- Use proper frontmatter (segment, tags) on new documents
- Keep notes concise and well-structured
- Link related documents
- Respect vault segments: agents/, personal/, work/, projects/, knowledge/

## Output
- What you found or created
- Paths of documents read/written
- Any connections or patterns discovered`,
  model: "haiku" as const,
  tools: [
    "Read", "Glob", "Grep",
    ...MCP_TOOLS,
  ],
  mcpServers: ["openclaw-tools"] as any[],
  maxTurns: 15,
};

/** Config for orchestrator prompt — describes when to use memory agent vs direct MCP tools */
export const memoryAgentConfig = {
  mcpTools: MCP_TOOLS,
  description:
    "For simple vault lookups, use MCP tools directly (mem_search, mem_get, tag_search). " +
    "For complex knowledge tasks (vault reorganization, multi-source synthesis, knowledge graph operations), " +
    "spawn the memory agent (Haiku, cost-effective).",
};
