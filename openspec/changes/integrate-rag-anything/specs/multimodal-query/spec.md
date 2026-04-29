## ADDED Requirements

### Requirement: Image Query
The system SHALL support querying with image content for VLM-enhanced analysis.

#### Scenario: Query with image data
- **WHEN** user calls rag.aquery_with_multimodal(query, multimodal_content=[{"type": "image", ...}])
- **THEN** system uses VLM to analyze image and returns answer

#### Scenario: Auto VLM enhancement
- **WHEN** user provides vision_model_func and queries about images in document
- **THEN** system automatically enhances with VLM analysis

### Requirement: Table Query
The system SHALL support querying with table content for comparison.

#### Scenario: Query comparing with table data
- **WHEN** user calls rag.aquery_with_multimodal(query, multimodal_content=[{"type": "table", ...}])
- **THEN** system compares query with provided table data

### Requirement: Equation Query
The system SHALL support querying with LaTeX formula content.

#### Scenario: Query about formula
- **WHEN** user calls rag.aquery_with_multimodal(query, multimodal_content=[{"type": "equation", "latex": "..."}])
- **THEN** system explains formula in context of knowledge base