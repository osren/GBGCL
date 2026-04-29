## ADDED Requirements

### Requirement: Batch Folder Processing
The system SHALL support processing multiple documents from a folder directory.

#### Scenario: Process all PDFs in directory
- **WHEN** user calls rag.process_folder_complete(folder_path="./docs")
- **THEN** all PDF files in the folder are parsed and stored

#### Scenario: Filter by file extension
- **WHEN** user specifies file_extensions=[".pdf", ".docx"]
- **THEN** only files with those extensions are processed

#### Scenario: Recursive processing
- **WHEN** user sets recursive=True
- **THEN** subdirectories are also processed

#### Scenario: Limit workers
- **WHEN** user specifies max_workers=4
- **THEN** at most 4 documents are processed concurrently