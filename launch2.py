from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional
import yaml
import json
import logging
from enum import Enum
from datetime import datetime
from llm_providers import create_llm_provider

class TransformationStep(Enum):
    MISSION_TO_STORY = "mission_to_story"
    STORY_TO_GHERKIN = "story_to_gherkin"
    GHERKIN_TO_DSL = "gherkin_to_dsl"
    DSL_TO_MVVM = "dsl_to_mvvm"

@dataclass
class WorkflowStep:
    name: str
    actions: List[str]
    transitions: List[str]
    
    def to_dict(self):
        return {
            "name": self.name,
            "actions": self.actions,
            "transitions": self.transitions
        }

@dataclass
class Story:
    name: str
    actor: str
    workflow_steps: List[WorkflowStep]
    
    def to_dict(self):
        return {
            "name": self.name,
            "actor": self.actor,
            "workflow_steps": [step.to_dict() for step in self.workflow_steps]
        }

@dataclass
class TraceabilityRecord:
    timestamp: str
    source_file: str
    target_file: str
    transformation_type: TransformationStep
    validation_status: str = "pending"
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "source_file": self.source_file,
            "target_file": self.target_file,
            "transformation_type": self.transformation_type.value,
            "validation_status": self.validation_status
        }

class DomainPipeline:
    def __init__(self, project_name: str, llm_provider: str = "ollama"):
        self.project_name = project_name
        self.base_path = Path(f"./projects/{project_name}")
        self.setup_project_structure()
        self.traceability_records: List[TraceabilityRecord] = []
        self.setup_logging()
        self.llm = create_llm_provider(llm_provider)

    def setup_logging(self):
        log_file = self.base_path / "launchpad" / "pipeline.log"
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_project_structure(self):
        """Create simplified project directory structure"""
        directories = [
            "launchpad",
            "features"
        ]
        
        for dir_name in directories:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)

    def save_traceability(self):
        """Save traceability records to JSON"""
        trace_file = self.base_path / "launchpad" / "traceability.json"
        records_dict = [record.to_dict() for record in self.traceability_records]
        
        with open(trace_file, 'w') as f:
            json.dump(records_dict, f, indent=2)

    def llm_transform(self, prompt: str) -> str:
        """Transform content using configured LLM provider"""
        try:
            logging.info(f"LLM Prompt:\n{prompt}\n")
            response = self.llm.generate(prompt)
            logging.info(f"LLM Response:\n{response}\n")
            return response
        except Exception as e:
            logging.error(f"LLM transformation failed: {str(e)}")
            raise

    def parse_structured_content(self, content: str) -> List[Story]:
        """Parse LLM-structured content into Story objects"""
        # Example implementation - adjust based on actual LLM output format
        return [
            Story(
                name="Shopping Cart Workflow",
                actor="Customer",
                workflow_steps=[
                    WorkflowStep(
                        name="Browse Products",
                        actions=["View catalog", "Search for products"],
                        transitions=["Can proceed to Add to Cart"]
                    ),
                    WorkflowStep(
                        name="Add to Cart",
                        actions=["Select quantity", "Add item to cart"],
                        transitions=["Can proceed to Review Cart"]
                    )
                ]
            )
        ]

    def serialize_stories_to_yaml(self, stories: List[Story]) -> str:
        """Convert stories to YAML format without Python object tags"""
        stories_dict = {"stories": [story.to_dict() for story in stories]}
        return yaml.dump(stories_dict, sort_keys=False, default_flow_style=False,
                        explicit_start=True, allow_unicode=True,
                        Dumper=yaml.SafeDumper)

    def generate_mvvm_implementation(self, dsl_content: dict) -> Dict[str, str]:
        """Generate MVVM implementation from DSL"""
        logging.info("Starting MVVM implementation generation...")
        
        # Extract domain concepts from DSL and store workflow states
        logging.info("Extracting workflow states from DSL...")
        workflow_states = self._extract_workflow_states(dsl_content)
        logging.info(f"Found {len(workflow_states)} workflow states")
        
        logging.info("Extracting domain entities from DSL...")
        domain_entities = self._extract_domain_entities(dsl_content)
        logging.info(f"Found {len(domain_entities)} domain entities")
        
        # Generate code for each component
        logging.info("Generating domain models...")
        models = self._generate_domain_models(domain_entities)
        
        logging.info("Generating viewmodels...")
        viewmodels = {}
        for state in workflow_states:
            vm_name = f"viewmodel_{state['name'].lower()}.py"
            viewmodels[vm_name] = self._generate_viewmodel(state)
            logging.info(f"Generated viewmodel for {state['name']}")
        
        logging.info("Generating API endpoints...")
        api = self._generate_api_endpoints(workflow_states)
        
        # Prepare return dictionary with all generated files
        generated_files = {
            "models/models.py": models,
            "api/api.py": api,
            **{f"viewmodels/{k}": v for k, v in viewmodels.items()}
        }
        
        logging.info(f"Generated {len(generated_files)} MVVM implementation files")
        return generated_files

    def _extract_domain_entities(self, dsl: dict) -> List[dict]:
        """Extract domain entities and their properties from DSL"""
        entities = []
        if "domain" in dsl:
            for entity in dsl["domain"].get("entities", []):
                properties = []
                for prop in entity.get("properties", []):
                    prop_type = self._map_dsl_type_to_python(prop.get("type", "str"))
                    properties.append({
                        "name": prop["name"],
                        "type": prop_type,
                        "required": prop.get("required", True)
                    })
                entities.append({
                    "name": entity["name"],
                    "properties": properties,
                    "behaviors": entity.get("behaviors", []),
                    "validations": entity.get("validations", [])
                })
        return entities

    def _map_dsl_type_to_python(self, dsl_type: str) -> str:
        """Map DSL types to Python types"""
        type_mapping = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "date": "datetime",
            "money": "Money",
            "array": "List",
            "object": "Dict",
        }
        return type_mapping.get(dsl_type.lower(), dsl_type)

    def _extract_workflow_states(self, dsl: dict) -> List[dict]:
        """Extract workflow states and transitions from DSL"""
        states = []
        if "workflows" in dsl:
            for workflow in dsl["workflows"]:
                states.append({
                    "name": workflow["name"],
                    "actions": workflow.get("actions", []),
                    "transitions": workflow.get("transitions", []),
                    "validations": workflow.get("validations", [])
                })
        self._workflow_states = states  # Store for later use
        return states

    def _generate_viewmodel(self, state: dict) -> str:
        """Generate a ViewModel class for a workflow state"""
        class_name = f"ViewModel{state['name']}"
        
        # Generate action methods
        action_methods = []
        for action in state.get('actions', []):
            params = [f"{p['name']}: {p['type']}" for p in action.get('parameters', [])]
            validations = [
                f"        if not {v['condition']}:\n            raise HTTPException(status_code=400, detail='{v['message']}')"
                for v in action.get('validations', [])
            ]
            
            action_methods.append(f'''
    async def {action["name"]}(self, session_id: UUID, {", ".join(params)}) -> dict:
        """Handle {action["name"]} action"""
        if session_id not in self._state_data:
            raise HTTPException(status_code=404, detail="Session not found")
            
        state_data = self._state_data[session_id]
        {chr(10).join(validations) if validations else "        # Perform action-specific logic"}
        
        return self._create_response(state_data)''')

        # Generate transition methods
        transition_methods = []
        for transition in state.get('transitions', []):
            next_state = transition['target_state']
            conditions = [
                f"        if not {c['condition']}:\n            raise HTTPException(status_code=400, detail='{c['message']}')"
                for c in transition.get('conditions', [])
            ]
            
            transition_methods.append(f'''
    async def proceed_to_{next_state.lower()}(self, session_id: UUID) -> dict:
        """Transition to {next_state} state"""
        if session_id not in self._state_data:
            raise HTTPException(status_code=404, detail="Session not found")
            
        state_data = self._state_data[session_id]
        {chr(10).join(conditions) if conditions else "        # Validate transition conditions"}
        
        return {{
            "workflow_state": WorkflowState.{next_state.upper()},
            "next_url": f"/api/{next_state.lower()}/{{session_id}}",
            "method": "POST"
        }}''')

        # Generate the complete viewmodel class
        return f'''
class {class_name}:
    """Handles the {state["name"].lower()} phase of the workflow"""
    def __init__(self):
        self._state_data: Dict[UUID, Dict] = {{}}
        
    {chr(10).join(action_methods)}
    {chr(10).join(transition_methods)}
    
    def _create_response(self, data: dict) -> dict:
        """Create HATEOAS response"""
        return {{
            "workflow_state": WorkflowState.{state["name"].upper()},
            "data": data,
            "available_actions": self._get_available_actions(data)
        }}
        
    def _get_available_actions(self, data: dict) -> dict:
        """Get available actions based on current state"""
        actions = {{}}
        
        # Add action endpoints
        for action in {state.get("actions", [])}:
            action_name = action["name"]
            actions[action_name] = {{
                "href": f"/api/{state['name'].lower()}/{{data['id']}}/{{action_name}}",
                "method": "POST",
                "required_fields": [p["name"] for p in action.get("parameters", [])]
            }}
            
        # Add transition endpoints
        for transition in {state.get("transitions", [])}:
            next_state = transition["target_state"]
            actions[f"proceed_to_{next_state.lower()}"] = {{
                "href": f"/api/{state['name'].lower()}/{{data['id']}}/transitions/{{next_state.lower()}}",
                "method": "POST",
                "conditions": [c["name"] for c in transition.get("conditions", [])]
            }}
            
        return actions'''

    def _generate_domain_models(self, entities: List[dict]) -> str:
        """Generate domain model classes"""
        imports = '''from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
from decimal import Decimal'''

        value_objects = '''
# Value Objects
@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str = "USD"'''

        # Generate entity classes
        entity_classes = []
        for entity in entities:
            props = []
            for prop in entity["properties"]:
                type_str = prop["type"]
                if not prop["required"]:
                    type_str = f"Optional[{type_str}]"
                props.append(f"{prop['name']}: {type_str}")
            
            validations = [f"    @property\n    def is_valid_{v['name']}(self) -> bool:\n        return {v['condition']}"
                          for v in entity["validations"]]
            
            behaviors = [f"    def {b['name']}(self, {b.get('params', '')}) -> {b.get('return_type', 'None')}:\n        {b['implementation']}"
                        for b in entity["behaviors"]]
            
            entity_classes.append(f'''
@dataclass
class {entity["name"]}:
    id: UUID = field(default_factory=uuid4)
    {chr(10).join(f"    {p}" for p in props)}
    
    {chr(10).join(validations) if validations else "    pass"}
    
    {chr(10).join(behaviors) if behaviors else ""}''')

        # Generate workflow states
        workflow_states = "\n    ".join(
            f"{s['name'].upper()} = '{s['name'].lower()}'" 
            for s in self._workflow_states
        )
        
        workflow_enum = f'''
# Workflow States
class WorkflowState(str, Enum):
    {workflow_states}'''

        return f"{imports}\n{value_objects}\n\n# Domain Entities{''.join(entity_classes)}\n{workflow_enum}"

    def _generate_action_method(self, action: str) -> str:
        """Generate an async action method for a viewmodel"""
        method_name = action.lower().replace(" ", "_")
        return f'''
    async def {method_name}(self, session_id: UUID, **kwargs) -> dict:
        """Handle {action} action"""
        if session_id not in self._state_data:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Update state data based on action
        state_data = self._state_data[session_id]
        # TODO: Implement action-specific logic
        
        return self._create_response(state_data)'''

    def _generate_transition_method(self, transition: str) -> str:
        """Generate a transition method for a viewmodel"""
        next_state = transition.split("Can proceed to")[1].strip()
        method_name = f"proceed_to_{next_state.lower().replace(' ', '_')}"
        return f'''
    async def {method_name}(self, session_id: UUID) -> dict:
        """Transition to {next_state} state"""
        if session_id not in self._state_data:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Validate transition conditions
        state_data = self._state_data[session_id]
        # TODO: Add transition-specific validation
        
        return {{
            "workflow_state": WorkflowState.{next_state.upper()},
            "next_url": f"/api/{next_state.lower()}/{{session_id}}",
            "method": "POST"
        }}'''

    def _generate_viewmodels(self, states: List[dict]) -> str:
        """Generate viewmodel classes"""
        imports = '''from typing import Dict, List
from uuid import UUID
from fastapi import HTTPException
from .models import *'''

        viewmodel_classes = []
        for state in states:
            # Generate action methods
            actions = []
            for action in state.get("actions", []):
                params = [f"{p['name']}: {p['type']}" for p in action.get("parameters", [])]
                validations = [f"        if not {v['condition']}:\n            raise HTTPException(status_code=400, detail='{v['message']}')"
                             for v in action.get("validations", [])]
                
                actions.append(f'''
    async def {action["name"]}(self, session_id: UUID, {", ".join(params)}) -> dict:
        """Handle {action["name"]} action"""
        if session_id not in self._state_data:
            raise HTTPException(status_code=404, detail="Session not found")
            
        state_data = self._state_data[session_id]
        {chr(10).join(validations) if validations else "        # Perform action-specific logic"}
        
        return self._create_response(state_data)''')

            # Generate transition methods
            transitions = []
            for transition in state.get("transitions", []):
                next_state = transition["target_state"]
                conditions = [f"        if not {c['condition']}:\n            raise HTTPException(status_code=400, detail='{c['message']}')"
                            for c in transition.get("conditions", [])]
                
                transitions.append(f'''
    async def proceed_to_{next_state.lower()}(self, session_id: UUID) -> dict:
        """Transition to {next_state} state"""
        if session_id not in self._state_data:
            raise HTTPException(status_code=404, detail="Session not found")
            
        state_data = self._state_data[session_id]
        {chr(10).join(conditions) if conditions else "        # Validate transition conditions"}
        
        return {{
            "workflow_state": WorkflowState.{next_state.upper()},
            "next_url": f"/api/{next_state.lower()}/{{session_id}}",
            "method": "POST"
        }}''')

            # Generate the complete viewmodel class
            viewmodel_classes.append(f'''
class {state["name"]}ViewModel:
    """Handles the {state["name"].lower()} phase of the workflow"""
    def __init__(self):
        self._state_data: Dict[UUID, Dict] = {{}}
        
    {chr(10).join(actions)}
    {chr(10).join(transitions)}
    
    def _create_response(self, data: dict) -> dict:
        """Create HATEOAS response"""
        return {{
            "workflow_state": WorkflowState.{state["name"].upper()},
            "data": data,
            "available_actions": self._get_available_actions(data)
        }}
        
    def _get_available_actions(self, data: dict) -> dict:
        """Get available actions based on current state"""
        actions = {{}}
        
        # Add action endpoints
        {self._generate_action_links(state)}
        
        # Add transition endpoints
        {self._generate_transition_links(state)}
        
        return actions''')

        return f"{imports}\n{''.join(viewmodel_classes)}"

    def _generate_action_links(self, state: dict) -> str:
        """Generate HATEOAS links for actions"""
        return f'''for action in {state.get("actions", [])}:
            action_name = action["name"]
            actions[action_name] = {{
                "href": f"/api/{state['name'].lower()}/{{data['id']}}/{{action_name}}",
                "method": "POST",
                "required_fields": [p["name"] for p in action.get("parameters", [])]
            }}'''

    def _generate_transition_links(self, state: dict) -> str:
        """Generate HATEOAS links for transitions"""
        actions = {}
        for transition in state.get("transitions", []):
            next_state = transition["target_state"]
            actions[f"proceed_to_{next_state.lower()}"] = {
                "href": f"/api/{state['name'].lower()}/{{data_id}}/transitions/{next_state.lower()}",
                "method": "POST",
                "conditions": [c["name"] for c in transition.get("conditions", [])]
            }
        return str(actions)

    def _generate_api_endpoints(self, states: List[dict]) -> str:
        """Generate FastAPI endpoints"""
        template = '''
from fastapi import FastAPI, HTTPException
from uuid import UUID
from .viewmodels import *

app = FastAPI()

{endpoints}
'''
        # Generate endpoints for each workflow state
        endpoints = []
        for state in states:
            for action in state['actions']:
                param_list = [f"{p['name']}: {p['type']}" for p in action.get('parameters', [])]
                param_names = [p['name'] for p in action.get('parameters', [])]
                
                endpoints.append(f'''
@app.post("/api/{state['name'].lower()}/{action['name'].lower()}")
async def {action['name'].lower()}(
    {", ".join(param_list)}
):
    """Handle {action['name']} action"""
    return await {state['name'].lower()}_vm.{action['name'].lower()}({
        ", ".join(param_names)
    })''')
                
        return template.format(endpoints="\n".join(endpoints))

    def _needs_processing(self, input_path: Path, output_path: Path) -> bool:
        """Check if processing is needed based on file modification times"""
        if not output_path.exists():
            return True
            
        input_mtime = input_path.stat().st_mtime
        output_mtime = output_path.stat().st_mtime
        
        return input_mtime > output_mtime

    def process_step(self, step: TransformationStep, input_file: str) -> None:
        """Process a single transformation step"""
        timestamp = datetime.now().isoformat()
        
        try:
            if step == TransformationStep.MISSION_TO_STORY:
                input_path = self.base_path / "launchpad" / input_file
                output_file = "story.yaml"
                output_path = self.base_path / "launchpad" / output_file
                
                if not self._needs_processing(input_path, output_path):
                    logging.info(f"Skipping {step.value} - {output_file} is up to date")
                    return
                    
                with open(input_path) as f:
                    mission_content = f.read()
                
                stories = self.mission_to_structured_format(mission_content)
                yaml_content = self.serialize_stories_to_yaml(stories)
                
                with open(output_path, 'w') as f:
                    f.write(yaml_content)
                
                record = TraceabilityRecord(
                    timestamp=timestamp,
                    source_file=input_file,
                    target_file=output_file,
                    transformation_type=step,
                    validation_status="success"
                )
                
                self.traceability_records.append(record)
                self.save_traceability()
                logging.info(f"Successfully processed {step.value} for {input_file}")
            
            # ... rest of the method remains the same ...
            
        except Exception as e:
            logging.error(f"Error processing {step.value} for {input_file}: {str(e)}")
            raise

    def mission_to_structured_format(self, mission_content: str) -> List[Story]:
        """Convert high-level mission to structured format using LLM"""
        prompt = f"""
Convert this high-level mission description into a structured format with clear workflows.
For each workflow, identify the actor, steps, actions, and transitions.
Format it as a clear list of workflows with their components.

Mission:
{mission_content}

Please structure it similar to this example:
Workflow: Shopping Cart
Actor: Customer
Steps:
1. Browse Products
   Actions:
   - View catalog
   - Search items
   Transitions:
   - Can add to cart

2. Add to Cart
   Actions:
   - Select quantity
   Transitions:
   - Can review cart
"""
        
        structured_content = self.llm_transform(prompt)
        stories = self.parse_structured_content(structured_content)
        return stories

    def parse_structured_content(self, content: str) -> List[Story]:
        """Parse LLM-structured content into Story objects"""
        stories = []
        current_workflow = None
        current_actor = None
        current_steps = []
        current_step = None
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Workflow:'):
                # Save previous workflow if exists
                if current_workflow:
                    stories.append(Story(
                        name=current_workflow,
                        actor=current_actor,
                        workflow_steps=current_steps
                    ))
                # Start new workflow
                current_workflow = line.split('Workflow:')[1].strip()
                current_steps = []
                
            elif line.startswith('Actor:'):
                current_actor = line.split('Actor:')[1].strip()
                
            elif line.strip().isdigit() or line[0].isdigit() and line[1] == '.':
                # Save previous step if exists
                if current_step:
                    current_steps.append(current_step)
                # Start new step
                step_name = line.split('.')[1].strip()
                current_step = WorkflowStep(
                    name=step_name,
                    actions=[],
                    transitions=[]
                )
                
            elif line.startswith('Actions:'):
                continue
            elif line.startswith('- ') and current_step:
                if 'Transitions:' in line:
                    continue
                if 'Can' in line:
                    current_step.transitions.append(line.strip('- '))
                else:
                    current_step.actions.append(line.strip('- '))
                    
        # Add final step and workflow
        if current_step:
            current_steps.append(current_step)
        if current_workflow:
            stories.append(Story(
                name=current_workflow,
                actor=current_actor,
                workflow_steps=current_steps
            ))
            
        return stories

    def enhance_story_to_gherkin(self, story_yaml: str) -> List[str]:
        """Convert YAML stories to Gherkin with LLM enhancement"""
        prompt = f"""
You are a Gherkin feature file generator. Output only valid Gherkin syntax.
Convert these YAML stories into MULTIPLE Gherkin feature files.
Create a separate Feature for each workflow.
Add detailed steps, validations, and error cases.

Input YAML:
{story_yaml}

For each workflow, create a separate feature like this:

Feature: [Workflow Name]
  As a [Actor]
  I want to [achieve something]
  So that [business value]

  Background:
    Given [any common preconditions]

  Scenario: Happy Path
    [Steps...]

  Scenario: Error Cases
    [Steps...]

# Start writing multiple Features, one per workflow
"""
        
        enhanced_scenarios = self.llm_transform(prompt)
        return self.parse_gherkin_content(enhanced_scenarios)

    def parse_gherkin_content(self, content: str) -> List[tuple]:
        """Parse LLM-generated Gherkin content into feature files"""
        features = []
        current_feature = []
        current_name = None
        
        for line in content.split('\n'):
            if line.startswith('Feature:'):
                # Save previous feature if exists
                if current_feature and current_name:
                    features.append((f"{current_name.lower().replace(' ', '_')}.feature", 
                                   '\n'.join(current_feature)))
                # Start new feature
                current_name = line.split('Feature:')[1].strip()
                current_feature = [line]
            else:
                if current_feature is not None:  # Only append if we've started a feature
                    current_feature.append(line)
        
        # Add the last feature
        if current_feature and current_name:
            features.append((f"{current_name.lower().replace(' ', '_')}.feature", 
                           '\n'.join(current_feature)))
        
        return features

    def generate_dsl_with_rules(self, feature_files: List[tuple], story_yaml: str) -> Dict:
        """Convert features and stories to DSL with additional business rules"""
        # Parse story YAML to extract workflow information
        story_data = yaml.safe_load(story_yaml)
        
        # Initialize DSL structure
        dsl = {
            "domain": {
                "entities": [],
                "value_objects": []
            },
            "workflows": []
        }
        
        # Extract entities and their properties from Gherkin features
        entities = self._extract_entities_from_features(feature_files)
        dsl["domain"]["entities"].extend(entities)
        
        # Add common value objects
        dsl["domain"]["value_objects"].extend([
            {
                "name": "Money",
                "properties": [
                    {"name": "amount", "type": "decimal", "required": True},
                    {"name": "currency", "type": "string", "required": True}
                ]
            }
        ])
        
        # Convert stories to workflows
        if "stories" in story_data:
            for story in story_data["stories"]:
                workflow = self._convert_story_to_workflow(story)
                dsl["workflows"].append(workflow)
        
        return dsl

    def _extract_entities_from_features(self, feature_files: List[tuple]) -> List[dict]:
        """Extract domain entities from Gherkin features"""
        entities = {}
        
        for _, content in feature_files:
            lines = content.split('\n')
            current_entity = None
            
            for line in lines:
                # Look for entity mentions in Given/When/Then steps
                if any(keyword in line for keyword in ['Given', 'When', 'Then']):
                    # Extract nouns that could be entities
                    words = line.split()
                    for word in words:
                        # Simple heuristic: capitalize words are likely entities
                        if word[0].isupper() and len(word) > 1:
                            entity_name = word.strip('.,')
                            if entity_name not in entities:
                                entities[entity_name] = {
                                    "name": entity_name,
                                    "properties": [],
                                    "behaviors": [],
                                    "validations": []
                                }
                
                # Look for properties in And steps
                if current_entity and 'And' in line:
                    # Extract property names and types
                    if 'with' in line or 'has' in line:
                        prop_match = line.lower().split('with')[-1].strip()
                        prop_name = prop_match.split()[0]
                        prop_type = self._infer_property_type(prop_match)
                        
                        entities[current_entity]["properties"].append({
                            "name": prop_name,
                            "type": prop_type,
                            "required": True
                        })
        
        return list(entities.values())

    def _infer_property_type(self, text: str) -> str:
        """Infer property type from text description"""
        text = text.lower()
        if any(word in text for word in ['amount', 'price', 'total']):
            return 'decimal'
        elif any(word in text for word in ['date', 'time']):
            return 'datetime'
        elif any(word in text for word in ['count', 'quantity', 'number']):
            return 'integer'
        elif any(word in text for word in ['is', 'has']):
            return 'boolean'
        else:
            return 'string'

    def _convert_story_to_workflow(self, story: dict) -> dict:
        """Convert a story to a workflow definition"""
        workflow = {
            "name": story["name"],
            "actor": story["actor"],
            "actions": [],
            "transitions": []
        }
        
        for step in story["workflow_steps"]:
            # Add actions
            for action in step["actions"]:
                workflow["actions"].append({
                    "name": action.lower().replace(" ", "_"),
                    "parameters": self._infer_action_parameters(action),
                    "validations": [{
                        "name": "basic_validation",
                        "condition": "all_required_fields_present",
                        "message": "All required fields must be provided"
                    }]
                })
            
            # Add transitions
            for transition in step["transitions"]:
                if "Can proceed to" in transition:
                    target_state = transition.split("Can proceed to")[1].strip()
                    workflow["transitions"].append({
                        "target_state": target_state,
                        "conditions": [{
                            "name": "can_transition",
                            "condition": f"current_state_valid_for_{target_state.lower()}",
                            "message": f"Cannot transition to {target_state} from current state"
                        }]
                    })
        
        return workflow

    def _infer_action_parameters(self, action: str) -> List[dict]:
        """Infer parameters needed for an action"""
        parameters = []
        words = action.lower().split()
        
        # Common parameter patterns
        if any(word in words for word in ['add', 'create', 'update']):
            parameters.append({"name": "data", "type": "object", "required": True})
        if any(word in words for word in ['id', 'identifier']):
            parameters.append({"name": "id", "type": "string", "required": True})
        if 'quantity' in words:
            parameters.append({"name": "quantity", "type": "integer", "required": True})
        if any(word in words for word in ['amount', 'price']):
            parameters.append({"name": "amount", "type": "decimal", "required": True})
            
        return parameters

    def process_step(self, step: TransformationStep, input_file: str) -> None:
        """Process a single transformation step"""
        timestamp = datetime.now().isoformat()
        
        try:
            if step == TransformationStep.MISSION_TO_STORY:
                with open(self.base_path / "launchpad" / input_file) as f:
                    mission_content = f.read()
                
                stories = self.mission_to_structured_format(mission_content)
                yaml_content = self.serialize_stories_to_yaml(stories)
                output_file = "story.yaml"
                
                with open(self.base_path / "launchpad" / output_file, 'w') as f:
                    f.write(yaml_content)
                
                record = TraceabilityRecord(
                    timestamp=timestamp,
                    source_file=input_file,
                    target_file=output_file,
                    transformation_type=step,
                    validation_status="success"
                )
                
            elif step == TransformationStep.STORY_TO_GHERKIN:
                input_path = self.base_path / "launchpad" / input_file
                features_dir = self.base_path / "features"
                
                # Check if any existing feature files are older than the story file
                needs_update = True
                if features_dir.exists() and list(features_dir.glob("*.feature")):
                    needs_update = any(
                        self._needs_processing(input_path, feature_path)
                        for feature_path in features_dir.glob("*.feature")
                    )
                
                if not needs_update:
                    logging.info(f"Skipping {step.value} - feature files are up to date")
                    return
                
                with open(input_path) as f:
                    story_yaml = f.read()
                
                feature_files = self.enhance_story_to_gherkin(story_yaml)
                
                for feature_file, content in feature_files:
                    with open(features_dir / feature_file, 'w') as f:
                        f.write(content)
                    
                    record = TraceabilityRecord(
                        timestamp=timestamp,
                        source_file=input_file,
                        target_file=feature_file,
                        transformation_type=step,
                        validation_status="success"
                    )
                    
            elif step == TransformationStep.GHERKIN_TO_DSL:
                features_dir = self.base_path / "features"
                output_file = "dsl.domain.yaml"
                output_path = self.base_path / "launchpad" / output_file
                
                # Check if either source files are newer than the DSL file
                story_file = self.base_path / "launchpad" / "story.yaml"
                needs_update = True
                if output_path.exists():
                    needs_update = (
                        any(self._needs_processing(feature_path, output_path)
                            for feature_path in features_dir.glob("*.feature")) or
                        self._needs_processing(story_file, output_path)
                    )
                
                if not needs_update:
                    logging.info(f"Skipping {step.value} - {output_file} is up to date")
                    return
                
                # Read story.yaml
                story_file = self.base_path / "launchpad" / "story.yaml"
                with open(story_file) as f:
                    story_yaml = f.read()

                # Read feature files
                feature_files = []
                for feature_file in features_dir.glob("*.feature"):
                    with open(feature_file) as f:
                        content = f.read()
                        feature_files.append((feature_file.name, content))
                
                dsl_content = self.generate_dsl_with_rules(feature_files, story_yaml)
                
                with open(output_path, 'w') as f:
                    yaml.dump(dsl_content, f, sort_keys=False)
                
                record = TraceabilityRecord(
                    timestamp=timestamp,
                    source_file=input_file,
                    target_file=output_file,
                    transformation_type=step,
                    validation_status="success"
                )

            elif step == TransformationStep.DSL_TO_MVVM:
                dsl_file = self.base_path / "launchpad" / "dsl.domain.yaml"
                mvvm_dir = self.base_path / "mvvm"
                
                # Check if any MVVM files need updating
                needs_update = True
                if mvvm_dir.exists():
                    mvvm_files = list(mvvm_dir.glob("*.py"))
                    if mvvm_files:
                        needs_update = any(
                            self._needs_processing(dsl_file, mvvm_file)
                            for mvvm_file in mvvm_files
                        )
                
                if not needs_update:
                    logging.info(f"Skipping {step.value} - MVVM files are up to date")
                    return
                
                with open(dsl_file) as f:
                    dsl_content = yaml.safe_load(f)

                logging.info("Starting MVVM code generation from DSL...")
                
                # Generate MVVM implementation
                mvvm_files = self.generate_mvvm_implementation(dsl_content)
                
                # Create mvvm directory and subdirectories
                mvvm_dir.mkdir(exist_ok=True)
                (mvvm_dir / "models").mkdir(exist_ok=True)
                (mvvm_dir / "viewmodels").mkdir(exist_ok=True)
                (mvvm_dir / "api").mkdir(exist_ok=True)
                
                logging.info(f"Created MVVM directory structure at {mvvm_dir}")
                
                # Write generated files to appropriate directories
                for filename, content in mvvm_files.items():
                    # Extract just the base filename without directory prefix
                    base_filename = filename.split('/')[-1]
                    
                    if filename.startswith("models"):
                        filepath = mvvm_dir / "models" / base_filename
                        logging.info(f"Writing domain models to {filepath}")
                    elif filename.startswith("viewmodel"):
                        filepath = mvvm_dir / "viewmodels" / base_filename
                        logging.info(f"Writing viewmodel to {filepath}")
                    elif filename.startswith("api"):
                        filepath = mvvm_dir / "api" / base_filename
                        logging.info(f"Writing API endpoints to {filepath}")
                    else:
                        filepath = mvvm_dir / base_filename
                        logging.info(f"Writing {filename} to {filepath}")
                    
                    with open(filepath, 'w') as f:
                        f.write(content)
                        logging.info(f"Successfully wrote {len(content)} bytes to {filepath}")
                
                record = TraceabilityRecord(
                    timestamp=timestamp,
                    source_file=str(dsl_file),
                    target_file=str(mvvm_dir),
                    transformation_type=step,
                    validation_status="success"
                )
            
            self.traceability_records.append(record)
            self.save_traceability()
            logging.info(f"Successfully processed {step.value} for {input_file}")
            
        except Exception as e:
            logging.error(f"Error processing {step.value} for {input_file}: {str(e)}")
            raise

def main():
    # Ensure project directory exists
    project_dir = Path("./projects/ecommerce")
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create launchpad directory if it doesn't exist
    launchpad_dir = project_dir / "launchpad"
    launchpad_dir.mkdir(exist_ok=True)
    
    # Copy prime-directive.md to mission.md if it doesn't exist
    mission_file = launchpad_dir / "mission.md"
    if not mission_file.exists():
        with open("prime-directive.md") as src, open(mission_file, "w") as dst:
            dst.write(src.read())

    # Initialize pipeline
    # DO NOT REMOVE THIS COMMENT: openai , claude , ollama
    pipeline = DomainPipeline("ecommerce", llm_provider="openai")
    
    # Define pipeline steps and their input files
    pipeline_steps = [
        (TransformationStep.MISSION_TO_STORY, "mission.md"),
        (TransformationStep.STORY_TO_GHERKIN, "story.yaml"),
        (TransformationStep.GHERKIN_TO_DSL, "*.feature"),
        (TransformationStep.DSL_TO_MVVM, "dsl.domain.yaml")
    ]
    
    print("\nAvailable starting points:")
    for idx, (step, input_file) in enumerate(pipeline_steps, 1):
        print(f"{idx}. {step.value} (Start from {input_file})")
    print(f"{len(pipeline_steps) + 1}. Run full pipeline")
    
    try:
        choice = input("\nEnter the number of your starting point (1-5): ").strip()
        choice_idx = int(choice) - 1
        
        if not 0 <= choice_idx <= len(pipeline_steps):
            print(f"Invalid choice. Please run again and select 1-{len(pipeline_steps) + 1}.")
            return
            
        # If choice is last option, start from beginning, otherwise start from chosen step
        start_idx = 0 if choice_idx == len(pipeline_steps) else choice_idx
        
        # Execute pipeline steps from the chosen starting point
        for step, input_file in pipeline_steps[start_idx:]:
            pipeline.process_step(step, input_file)
            
        print("Pipeline completed successfully!")
            
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        logging.error(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    main()
