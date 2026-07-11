## [ERR-20260710-001] agent-browser

**Logged**: 2026-07-10T00:00:00+08:00
**Priority**: low
**Status**: pending
**Area**: infra

### Summary
Agent Browser skill was selected, but its CLI is not installed in this environment.

### Error
```
zsh:1: command not found: agent-browser
```

### Context
- Attempted to inspect the Hugging Face dataset page with `agent-browser`.
- Continued with Hugging Face's official API and repository endpoints.

### Suggested Fix
Install the CLI before using this skill, or check availability first and use the official API fallback.

### Metadata
- Reproducible: yes
- Related Files: none

---
