# local_llm_sdk/utils/

## Purpose
Shared utility functions and helpers used across the SDK. Currently minimal - placeholder for future common utilities.

## Contents
- `__init__.py` - Empty module initialization (currently no utilities implemented)

## Relationships
- **Parent**: Imported by other SDK modules (`client.py`, `agents/`, `tools/`) as needed
- **Purpose**: Avoid code duplication by centralizing common helpers
- **Future**: Will contain logging, validation, formatting, or other shared functionality

## Future Additions
Potential utilities to add here:
- JSON serialization helpers (handling numpy types, dates, etc.)
- Logging configuration and decorators
- Retry logic with exponential backoff (if generalized beyond client)
- Response formatting utilities
- Validation helpers

## Notes
This directory exists for organizational purposes and future extensibility. If you need to add a helper function used by multiple SDK modules, this is the place.