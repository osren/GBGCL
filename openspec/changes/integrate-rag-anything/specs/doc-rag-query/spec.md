## ADDED Requirements

### Requirement: RAG Service Initialization
The system SHALL provide a RAG service module that can be initialized with API credentials from environment variables.

#### Scenario: Initialize with valid API key
- **WHEN** user calls RAGService.init() with valid OPENAI_API_KEY
- **THEN** service initializes and creates working directory

#### Scenario: Initialize with missing API key
- **WHEN** user calls RAGService.init() without OPENAI_API_KEY
- **THEN** ValueError is raised with clear error message

### Requirement: Document Parsing
The system SHALL support parsing PDF, DOCX, PPT, PPTX documents and storing them in knowledge base.

#### Scenario: Parse a PDF document
- **WHEN** user calls rag.process_document_complete(file_path="xxx.pdf")
- **THEN** document is parsed with MinerU, content stored in rag_storage

#### Scenario: Parse with custom output directory
- **WHEN** user specifies output_dir parameter
- **THEN** parsed content is saved to specified directory

### Requirement: Text Query
The system SHALL support querying the knowledge base with text questions.

#### Scenario: Hybrid mode query
- **WHEN** user calls rag.aquery("question", mode="hybrid")
- **THEN** system returns relevant answer with context

#### Scenario: Local mode query
- **WHEN** user calls rag.aquery("question", mode="local")
- **THEN** system returns answer from local graph neighborhood

#### Scenario: Global mode query
- **WHEN** user calls rag.aquery("question", mode="global")
- **THEN** system returns answer using global search

### Requirement: Query Mode Options
The system SHALL support all four query modes: naive, local, global, hybrid.

#### Scenario: All modes return valid results
- **WHEN** user queries with any supported mode
- **THEN** system returns non-empty answer