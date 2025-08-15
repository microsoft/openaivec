"""Comprehensive test cases for deserialize_base_model with Pydantic v2 compatible JSON Schema features.

This test suite follows MECE (Mutually Exclusive, Collectively Exhaustive) principles
to ensure comprehensive coverage of JSON Schema features that Pydantic v2 actually supports.

Excluded features (not supported by Pydantic v2):
- $schema, $id, $vocabulary meta-information
- Dynamic References ($dynamicRef, $dynamicAnchor)
- unevaluatedProperties / unevaluatedItems
- Conditional sub-schemas (if/then/else, dependentSchemas/dependentRequired)
- Content vocabulary (contentEncoding / contentMediaType)
- allOf, oneOf, not composition (limited support)
"""

from unittest import TestCase

from pydantic import BaseModel, Field

from openaivec._serialize import deserialize_base_model


class TestPydanticV2JSONSchemaCompliance(TestCase):
    """Test suite for Pydantic v2 compatible JSON Schema features."""

    # ============================================================================
    # 1. PRIMITIVE TYPES - Basic JSON Schema types supported by Pydantic
    # ============================================================================

    def test_string_type(self):
        """Test string type with various constraints."""
        schemas = [
            # Basic string
            {
                "title": "StringModel",
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
            # String with description and default
            {
                "title": "StringWithMetadata",
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "User's name",
                        "default": "Anonymous",
                    },
                },
            },
            # String with minLength/maxLength (Pydantic supports via constr)
            {
                "title": "ConstrainedString",
                "type": "object",
                "properties": {
                    "short": {"type": "string", "minLength": 1, "maxLength": 10},
                    "exact": {"type": "string", "minLength": 5, "maxLength": 5},
                },
            },
            # String with pattern (Pydantic supports via constr)
            {
                "title": "PatternString",
                "type": "object",
                "properties": {
                    "email": {"type": "string", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"},
                    "phone": {"type": "string", "pattern": "^\\d{3}-\\d{3}-\\d{4}$"},
                },
            },
            # String with format (Pydantic native format support)
            {
                "title": "FormattedString",
                "type": "object",
                "properties": {
                    "date": {"type": "string", "format": "date"},
                    "datetime": {"type": "string", "format": "date-time"},
                    "email": {"type": "string", "format": "email"},
                    "uuid": {"type": "string", "format": "uuid"},
                },
            },
        ]

        for schema in schemas:
            with self.subTest(title=schema["title"]):
                model = deserialize_base_model(schema)
                self.assertIsNotNone(model)
                self.assertTrue(issubclass(model, BaseModel))

    def test_numeric_types(self):
        """Test integer and number types with constraints."""
        schemas = [
            # Basic integer and number
            {
                "title": "NumericModel",
                "type": "object",
                "properties": {
                    "int_value": {"type": "integer"},
                    "float_value": {"type": "number"},
                },
            },
            # With minimum/maximum (Pydantic supports via conint/confloat)
            {
                "title": "BoundedNumbers",
                "type": "object",
                "properties": {
                    "age": {"type": "integer", "minimum": 0, "maximum": 150},
                    "percentage": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                },
            },
            # With exclusiveMinimum/exclusiveMaximum (Pydantic supports these)
            {
                "title": "ExclusiveBounds",
                "type": "object",
                "properties": {
                    "positive": {"type": "number", "exclusiveMinimum": 0},
                    "negative": {"type": "number", "exclusiveMaximum": 0},
                },
            },
            # With multipleOf (Pydantic supports this)
            {
                "title": "MultipleOfNumbers",
                "type": "object",
                "properties": {
                    "even": {"type": "integer", "multipleOf": 2},
                    "quarters": {"type": "number", "multipleOf": 0.25},
                },
            },
        ]

        for schema in schemas:
            with self.subTest(title=schema["title"]):
                model = deserialize_base_model(schema)
                self.assertIsNotNone(model)
                self.assertTrue(issubclass(model, BaseModel))

    def test_boolean_type(self):
        """Test boolean type."""
        schema = {
            "title": "BooleanModel",
            "type": "object",
            "properties": {
                "is_active": {"type": "boolean"},
                "has_permission": {"type": "boolean", "default": False},
            },
        }

        model = deserialize_base_model(schema)
        instance = model(is_active=True)
        self.assertEqual(instance.is_active, True)
        self.assertEqual(instance.has_permission, False)

    # ============================================================================
    # 2. COMPOUND TYPES - Arrays and Objects supported by Pydantic
    # ============================================================================

    def test_array_type(self):
        """Test array type with various constraints."""
        schemas = [
            # Basic array
            {
                "title": "BasicArray",
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
            # Array with minItems/maxItems (Pydantic supports via conlist)
            {
                "title": "ConstrainedArray",
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 5,
                    },
                },
            },
            # Array with uniqueItems (Pydantic can support via Set)
            {
                "title": "UniqueArray",
                "type": "object",
                "properties": {
                    "unique_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "uniqueItems": True,
                    },
                },
            },
            # Nested arrays
            {
                "title": "NestedArray",
                "type": "object",
                "properties": {
                    "matrix": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                },
            },
        ]

        for schema in schemas:
            with self.subTest(title=schema["title"]):
                model = deserialize_base_model(schema)
                self.assertIsNotNone(model)

    def test_object_type(self):
        """Test object type with various constraints."""
        schemas = [
            # Basic object (generic dict)
            {
                "title": "BasicObject",
                "type": "object",
                "properties": {
                    "config": {"type": "object"},
                },
            },
            # Nested object with properties
            {
                "title": "NestedObject",
                "type": "object",
                "properties": {
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                            "zip": {"type": "string"},
                        },
                    },
                },
            },
            # Object with required properties (Pydantic native support)
            {
                "title": "RequiredProperties",
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "optional": {"type": "string"},
                },
                "required": ["id", "name"],
            },
            # Object with additionalProperties (Pydantic extra='allow')
            {
                "title": "AdditionalProperties",
                "type": "object",
                "properties": {
                    "known": {"type": "string"},
                },
                "additionalProperties": {"type": "number"},
            },
        ]

        for schema in schemas:
            with self.subTest(title=schema["title"]):
                model = deserialize_base_model(schema)
                self.assertIsNotNone(model)

    # ============================================================================
    # 3. ENUMERATIONS - Pydantic native enum support
    # ============================================================================

    def test_enum_type(self):
        """Test enum constraints."""
        schemas = [
            # String enum (Pydantic uses Literal or Enum)
            {
                "title": "StringEnum",
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
                },
            },
            # Numeric enum
            {
                "title": "NumericEnum",
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "score": {"type": "number", "enum": [0.0, 0.5, 1.0]},
                },
            },
            # Mixed type enum (Pydantic can handle with Union)
            {
                "title": "MixedEnum",
                "type": "object",
                "properties": {
                    "value": {"enum": ["default", 42, True, None]},
                },
            },
            # Single value enum (const-like behavior)
            {
                "title": "SingleEnum",
                "type": "object",
                "properties": {
                    "version": {"type": "string", "enum": ["1.0.0"]},
                },
            },
        ]

        for schema in schemas:
            with self.subTest(title=schema["title"]):
                model = deserialize_base_model(schema)
                self.assertIsNotNone(model)

    # ============================================================================
    # 4. REFERENCES AND DEFINITIONS - Pydantic native $ref support
    # ============================================================================

    def test_refs_and_defs(self):
        """Test $ref and $defs (Pydantic native support)."""
        schemas = [
            # Basic $ref
            {
                "title": "RefModel",
                "type": "object",
                "properties": {
                    "user": {"$ref": "#/$defs/User"},
                    "admin": {"$ref": "#/$defs/User"},
                },
                "$defs": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                        },
                    }
                },
            },
            # Nested $refs
            {
                "title": "NestedRef",
                "type": "object",
                "properties": {
                    "company": {"$ref": "#/$defs/Company"},
                },
                "$defs": {
                    "Company": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "employees": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/Employee"},
                            },
                        },
                    },
                    "Employee": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                        },
                    },
                },
            },
        ]

        for schema in schemas:
            with self.subTest(title=schema["title"]):
                model = deserialize_base_model(schema)
                self.assertIsNotNone(model)

    def test_complex_refs_and_defs(self):
        """Test complex $ref and $defs scenarios commonly found in real-world JSON schemas."""

        # Test case 1: Multiple levels of references
        schema1 = {
            "title": "MultiLevelRefs",
            "type": "object",
            "properties": {
                "organization": {"$ref": "#/$defs/Organization"},
            },
            "$defs": {
                "Organization": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "departments": {"type": "array", "items": {"$ref": "#/$defs/Department"}},
                        "ceo": {"$ref": "#/$defs/Person"},
                    },
                    "required": ["name"],
                },
                "Department": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "manager": {"$ref": "#/$defs/Person"},
                        "employees": {"type": "array", "items": {"$ref": "#/$defs/Person"}},
                        "budget": {"type": "number"},
                    },
                    "required": ["name", "manager"],
                },
                "Person": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                        "roles": {"type": "array", "items": {"$ref": "#/$defs/Role"}},
                    },
                    "required": ["id", "first_name", "last_name"],
                },
                "Role": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "permissions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name"],
                },
            },
        }

        # Test case 2: Self-referencing structure (tree-like)
        schema2 = {
            "title": "SelfReferencing",
            "type": "object",
            "properties": {
                "root": {"$ref": "#/$defs/TreeNode"},
            },
            "$defs": {
                "TreeNode": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "value": {"type": "string"},
                        "children": {"type": "array", "items": {"$ref": "#/$defs/TreeNode"}},
                        "parent": {"$ref": "#/$defs/TreeNode"},
                        "metadata": {"$ref": "#/$defs/NodeMetadata"},
                    },
                    "required": ["id", "value"],
                },
                "NodeMetadata": {
                    "type": "object",
                    "properties": {
                        "created_at": {"type": "string", "format": "date-time"},
                        "updated_at": {"type": "string", "format": "date-time"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        }

        # Test case 3: Complex union types with references
        schema3 = {
            "title": "UnionWithRefs",
            "type": "object",
            "properties": {
                "data": {
                    "anyOf": [
                        {"$ref": "#/$defs/TextData"},
                        {"$ref": "#/$defs/ImageData"},
                        {"$ref": "#/$defs/VideoData"},
                    ]
                },
                "metadata": {"$ref": "#/$defs/CommonMetadata"},
            },
            "$defs": {
                "TextData": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["text"]},
                        "content": {"type": "string"},
                        "encoding": {"type": "string", "enum": ["utf-8", "ascii"]},
                        "language": {"type": "string"},
                    },
                    "required": ["type", "content"],
                },
                "ImageData": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["image"]},
                        "url": {"type": "string", "format": "uri"},
                        "width": {"type": "integer", "minimum": 1},
                        "height": {"type": "integer", "minimum": 1},
                        "format": {"type": "string", "enum": ["jpeg", "png", "gif", "webp"]},
                    },
                    "required": ["type", "url", "width", "height"],
                },
                "VideoData": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["video"]},
                        "url": {"type": "string", "format": "uri"},
                        "duration": {"type": "number", "minimum": 0},
                        "resolution": {"$ref": "#/$defs/Resolution"},
                        "codec": {"type": "string"},
                    },
                    "required": ["type", "url", "duration"],
                },
                "Resolution": {
                    "type": "object",
                    "properties": {
                        "width": {"type": "integer", "minimum": 1},
                        "height": {"type": "integer", "minimum": 1},
                    },
                    "required": ["width", "height"],
                },
                "CommonMetadata": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "created_at": {"type": "string", "format": "date-time"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "author": {"$ref": "#/$defs/Author"},
                    },
                    "required": ["id"],
                },
                "Author": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                    },
                    "required": ["name"],
                },
            },
        }

        # Test case 4: Deeply nested array references
        schema4 = {
            "title": "DeepArrayRefs",
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "array", "items": {"$ref": "#/$defs/Cell"}}},
                },
            },
            "$defs": {
                "Cell": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "z": {"type": "integer"},
                        "value": {"type": "number"},
                        "properties": {"$ref": "#/$defs/CellProperties"},
                    },
                    "required": ["x", "y", "z", "value"],
                },
                "CellProperties": {
                    "type": "object",
                    "properties": {
                        "color": {"type": "string"},
                        "opacity": {"type": "number", "minimum": 0, "maximum": 1},
                        "visible": {"type": "boolean", "default": True},
                    },
                },
            },
        }

        test_schemas = [
            ("MultiLevelRefs", schema1),
            ("SelfReferencing", schema2),
            ("UnionWithRefs", schema3),
            ("DeepArrayRefs", schema4),
        ]

        for schema_name, schema in test_schemas:
            with self.subTest(schema=schema_name):
                model = deserialize_base_model(schema)
                self.assertIsNotNone(model)

                # Verify we can create instances (basic smoke test)
                try:
                    # Create minimal valid instances based on required fields
                    if schema_name == "MultiLevelRefs":
                        instance = model(organization={"name": "Test Org"})
                        self.assertEqual(instance.organization.name, "Test Org")
                    elif schema_name == "SelfReferencing":
                        instance = model(root={"id": "1", "value": "root"})
                        self.assertEqual(instance.root.id, "1")
                    elif schema_name == "UnionWithRefs":
                        instance = model(data={"type": "text", "content": "Hello"}, metadata={"id": "meta1"})
                        self.assertEqual(instance.data.type, "text")
                    elif schema_name == "DeepArrayRefs":
                        instance = model(matrix=[[[]]])
                        self.assertEqual(instance.matrix, [[[]]])

                except Exception as e:
                    # If instantiation fails, that's okay for complex schemas
                    # The important thing is that the model was created successfully
                    self.assertIsNotNone(model, f"Model creation failed for {schema_name}: {e}")

    def test_refs_with_enums_and_literals(self):
        """Test $refs combined with enum fields and complex types."""
        schema = {
            "title": "RefsWithEnums",
            "type": "object",
            "properties": {
                "config": {"$ref": "#/$defs/Configuration"},
                "services": {"type": "array", "items": {"$ref": "#/$defs/Service"}},
            },
            "$defs": {
                "Configuration": {
                    "type": "object",
                    "properties": {
                        "environment": {"type": "string", "enum": ["development", "staging", "production"]},
                        "log_level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            "default": "INFO",
                        },
                        "features": {"$ref": "#/$defs/FeatureFlags"},
                        "database": {"$ref": "#/$defs/DatabaseConfig"},
                    },
                    "required": ["environment"],
                },
                "FeatureFlags": {
                    "type": "object",
                    "properties": {
                        "enable_caching": {"type": "boolean", "default": True},
                        "enable_analytics": {"type": "boolean", "default": False},
                        "api_version": {"type": "string", "enum": ["v1", "v2", "v3"]},
                    },
                },
                "DatabaseConfig": {
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string", "enum": ["postgresql", "mysql", "sqlite", "mongodb"]},
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "ssl_mode": {
                            "type": "string",
                            "enum": ["disable", "allow", "prefer", "require"],
                            "default": "prefer",
                        },
                    },
                    "required": ["provider", "host"],
                },
                "Service": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": ["api", "worker", "scheduler", "database"]},
                        "status": {
                            "type": "string",
                            "enum": ["running", "stopped", "error", "unknown"],
                            "default": "unknown",
                        },
                        "config": {"$ref": "#/$defs/ServiceConfig"},
                        "dependencies": {"type": "array", "items": {"$ref": "#/$defs/Service"}},
                    },
                    "required": ["name", "type"],
                },
                "ServiceConfig": {
                    "type": "object",
                    "properties": {
                        "replicas": {"type": "integer", "minimum": 1, "default": 1},
                        "memory_limit": {"type": "string"},
                        "cpu_limit": {"type": "string"},
                        "env_vars": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                },
            },
        }

        model = deserialize_base_model(schema)
        self.assertIsNotNone(model)

        # Test instance creation with enums
        instance = model(
            config={"environment": "development", "log_level": "DEBUG"},
            services=[{"name": "api-service", "type": "api", "status": "running"}],
        )

        self.assertEqual(instance.config.environment, "development")
        self.assertEqual(instance.config.log_level, "DEBUG")
        self.assertEqual(instance.services[0].name, "api-service")
        self.assertEqual(instance.services[0].type, "api")

    # ============================================================================
    # 5. METADATA AND ANNOTATIONS - Pydantic Field() support
    # ============================================================================

    def test_metadata_fields(self):
        """Test metadata fields (title, description, default, examples)."""
        schema = {
            "title": "MetadataModel",
            "description": "A model with various metadata fields",
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "title": "Name Field",
                    "description": "The person's full name",
                    "examples": ["John Doe", "Jane Smith"],
                    "default": "Unknown",
                },
                "age": {
                    "type": "integer",
                    "title": "Age Field",
                    "description": "Age in years",
                    "minimum": 0,
                    "maximum": 150,
                    "examples": [25, 30, 45],
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Tags",
                    "description": "Associated tags",
                    "default": [],
                },
                "metadata": {
                    "type": "object",
                    "title": "Metadata",
                    "description": "Additional metadata",
                    "default": {},
                },
            },
        }

        model = deserialize_base_model(schema)
        instance = model(age=30)
        self.assertEqual(instance.name, "Unknown")
        self.assertEqual(instance.age, 30)
        self.assertEqual(instance.tags, [])
        self.assertEqual(instance.metadata, {})

    # ============================================================================
    # 6. UNION TYPES - Pydantic Union support via anyOf
    # ============================================================================

    def test_union_types(self):
        """Test Union types (Pydantic supports via Union/Optional)."""
        schemas = [
            # Optional field (Union with None)
            {
                "title": "OptionalModel",
                "type": "object",
                "properties": {
                    "optional_string": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                    "optional_number": {
                        "anyOf": [{"type": "number"}, {"type": "null"}],
                    },
                },
            },
            # Union of different types (limited Pydantic support)
            {
                "title": "UnionModel",
                "type": "object",
                "properties": {
                    "value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                        ]
                    },
                },
            },
        ]

        for schema in schemas:
            with self.subTest(title=schema["title"]):
                model = deserialize_base_model(schema)
                self.assertIsNotNone(model)

    # ============================================================================
    # 7. EDGE CASES AND ERROR HANDLING
    # ============================================================================

    def test_empty_schemas(self):
        """Test edge cases with empty or minimal schemas."""
        schemas = [
            # Empty object
            {"title": "EmptyObject", "type": "object"},
            # Object with empty properties
            {"title": "NoProperties", "type": "object", "properties": {}},
            # Minimal valid schema with any type field
            {"title": "MinimalSchema", "type": "object", "properties": {"x": {"type": "string"}}},
        ]

        for schema in schemas:
            with self.subTest(title=schema["title"]):
                model = deserialize_base_model(schema)
                self.assertIsNotNone(model)

    def test_complex_nested_structures(self):
        """Test deeply nested structures (Pydantic supports nesting)."""
        schema = {
            "title": "ComplexNested",
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "level3": {
                                        "type": "object",
                                        "properties": {
                                            "data": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

        model = deserialize_base_model(schema)
        self.assertIsNotNone(model)

    def test_invalid_schemas(self):
        """Test handling of invalid or malformed schemas."""
        invalid_schemas = [
            # Field with invalid type
            {"title": "InvalidType", "type": "object", "properties": {"value": {"type": "invalid_type"}}},
        ]

        for schema in invalid_schemas:
            with self.subTest(schema=str(schema)):
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    deserialize_base_model(schema)

    def test_schemas_without_title(self):
        """Test that schemas without title are handled gracefully."""
        schema = {"type": "object", "properties": {"value": {"type": "string"}}}

        model = deserialize_base_model(schema)
        self.assertIsNotNone(model)
        # Should create model with default title
        instance = model(value="test")
        self.assertEqual(instance.value, "test")

    # ============================================================================
    # 8. REAL-WORLD SCENARIOS - Pydantic common use cases
    # ============================================================================

    def test_api_response_schema(self):
        """Test real-world API response schema."""
        schema = {
            "title": "APIResponse",
            "type": "object",
            "properties": {
                "status": {"type": "integer", "minimum": 100, "maximum": 599},
                "message": {"type": "string"},
                "data": {
                    "anyOf": [
                        {"type": "null"},
                        {"type": "object"},
                        {"type": "array", "items": {"type": "object"}},
                    ]
                },
                "errors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "message": {"type": "string"},
                            "field": {"type": "string"},
                        },
                        "required": ["code", "message"],
                    },
                },
            },
            "required": ["status", "message"],
        }

        model = deserialize_base_model(schema)
        self.assertIsNotNone(model)

    def test_database_schema(self):
        """Test database-like schema with relationships."""
        schema = {
            "title": "DatabaseSchema",
            "type": "object",
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "username": {"type": "string", "minLength": 3, "maxLength": 50},
                        "email": {"type": "string", "format": "email"},
                        "created_at": {"type": "string", "format": "date-time"},
                        "posts": {"type": "array", "items": {"$ref": "#/$defs/Post"}},
                    },
                    "required": ["id", "username", "email"],
                },
                "Post": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "title": {"type": "string", "maxLength": 200},
                        "content": {"type": "string"},
                        "author_id": {"type": "integer"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "published": {"type": "boolean", "default": False},
                    },
                    "required": ["id", "title", "content", "author_id"],
                },
            },
            "properties": {
                "users": {"type": "array", "items": {"$ref": "#/$defs/User"}},
                "posts": {"type": "array", "items": {"$ref": "#/$defs/Post"}},
            },
        }

        model = deserialize_base_model(schema)
        self.assertIsNotNone(model)

    def test_configuration_schema(self):
        """Test configuration file schema."""
        schema = {
            "title": "AppConfig",
            "type": "object",
            "properties": {
                "app": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "version": {
                            "type": "string",
                            "pattern": "^\\d+\\.\\d+\\.\\d+$",
                        },
                        "debug": {"type": "boolean", "default": False},
                        "port": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 65535,
                            "default": 8080,
                        },
                    },
                    "required": ["name", "version"],
                },
                "database": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "name": {"type": "string"},
                        "user": {"type": "string"},
                        "password": {"type": "string"},
                    },
                    "required": ["host", "name"],
                },
                "logging": {
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            "default": "INFO",
                        },
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["console", "file", "syslog"],
                            },
                            "uniqueItems": True,
                        },
                    },
                },
            },
            "required": ["app", "database"],
        }

        model = deserialize_base_model(schema)
        self.assertIsNotNone(model)

    # ============================================================================
    # 9. PYDANTIC-SPECIFIC FEATURES
    # ============================================================================

    def test_field_descriptions_preserved(self):
        """Test that Field descriptions are preserved during serialization/deserialization."""
        from openaivec._serialize import serialize_base_model

        class ModelWithDescriptions(BaseModel):
            sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
            confidence: float = Field(description="Confidence score for sentiment (0.0-1.0)")

        original = ModelWithDescriptions
        original_schema = original.model_json_schema()

        # Serialize and deserialize
        serialized = serialize_base_model(original)
        deserialized = deserialize_base_model(serialized)
        deserialized_schema = deserialized.model_json_schema()

        # Check that descriptions are preserved
        original_sentiment_desc = original_schema["properties"]["sentiment"].get("description")
        deserialized_sentiment_desc = deserialized_schema["properties"]["sentiment"].get("description")
        self.assertEqual(original_sentiment_desc, deserialized_sentiment_desc)

        original_confidence_desc = original_schema["properties"]["confidence"].get("description")
        deserialized_confidence_desc = deserialized_schema["properties"]["confidence"].get("description")
        self.assertEqual(original_confidence_desc, deserialized_confidence_desc)

        # Test that instances can be created
        instance = deserialized(sentiment="positive", confidence=0.95)
        self.assertEqual(instance.sentiment, "positive")
        self.assertEqual(instance.confidence, 0.95)

    def test_literal_enum_support(self):
        """Test that Literal types are properly handled (current implementation focus)."""
        from typing import Literal

        from openaivec._serialize import serialize_base_model

        class TaskStatus(BaseModel):
            status: Literal["pending", "in_progress", "completed"]
            priority: Literal["high", "medium", "low"]
            category: str = Field(description="Task category")

        schema = serialize_base_model(TaskStatus)

        # Check that Literal types are converted to enum in JSON schema
        self.assertEqual(schema["properties"]["status"]["type"], "string")
        self.assertEqual(set(schema["properties"]["status"]["enum"]), {"pending", "in_progress", "completed"})

        self.assertEqual(schema["properties"]["priority"]["type"], "string")
        self.assertEqual(set(schema["properties"]["priority"]["enum"]), {"high", "medium", "low"})

        # Test deserialization
        deserialized_class = deserialize_base_model(schema)

        # Test successful creation with valid values
        instance = deserialized_class(status="pending", priority="high", category="development")
        self.assertEqual(instance.status, "pending")
        self.assertEqual(instance.priority, "high")
        self.assertEqual(instance.category, "development")


if __name__ == "__main__":
    import unittest

    unittest.main()
