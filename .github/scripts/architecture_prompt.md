Update the ARCHITECTURE.md file for the kumo-rfm repository.

**Goal:** Create a concise, high-information-density architecture document
that helps engineers quickly understand this repo's structure, purpose,
and how it fits into the larger Kumo ecosystem.

**Repository Structure:**

```
{repo_structure}
```

**Key Files:**

README.md:

```
{readme}
```

**Instructions:**

Create ARCHITECTURE.md content with these sections:

- **Header:** Repo name + one-line description
- **Overview:** 2-3 sentences on purpose and primary consumers
- **Directory Structure:** Tree with descriptions (important dirs)
- **Key Components:** Table of apps/notebooks with descriptions
- **Integration Points:** Table of relationships with kumo-* repos
- **Entry Points:** Table for where to start for common tasks

**Style guidelines:**

- Maximize information density: prefer tables over paragraphs
- Be specific: name actual files/notebooks, not vague descriptions
- Keep total length under 100 lines if possible
- No fluff, no filler phrases

**Important:** Do NOT include the metadata header
(`<!--ARCHITECTURE_META...-->`). That will be added separately.
Start directly with `# kumo-rfm`

Output only the markdown content, nothing else.
