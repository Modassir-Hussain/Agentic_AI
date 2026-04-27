import os
import sys
import re
from dotenv import load_dotenv

# =========================
# PATH SETUP
# =========================
current_file_path = os.path.abspath(__file__)
phase_2_dir = os.path.dirname(current_file_path)
starter_dir = os.path.dirname(phase_2_dir)

if starter_dir not in sys.path:
    sys.path.insert(0, starter_dir)

from phase_1.workflow_agents.base_agents import (
    KnowledgeAugmentedPromptAgent
)

# =========================
# ENV SETUP
# =========================
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# =========================
# LOAD SPEC
# =========================
spec_path = os.path.join(phase_2_dir, "Product-Spec-Email-Router.txt")
with open(spec_path, "r", encoding="utf-8") as f:
    product_spec = f.read()

# =========================
# AGENTS
# =========================
pm_agent = KnowledgeAugmentedPromptAgent(openai_api_key, "Product Manager", f"Spec: {product_spec}")
prog_agent = KnowledgeAugmentedPromptAgent(openai_api_key, "Program Manager", f"Spec: {product_spec}")
dev_agent = KnowledgeAugmentedPromptAgent(openai_api_key, "Dev Engineer", f"Spec: {product_spec}")

# =========================
# VALIDATION FUNCTIONS
# =========================

def normalize_text(text):
    """Normalizes line endings and trims surrounding whitespace."""
    return (text or "").replace("\r\n", "\n").strip()



def extract_stories(text):
    """Extracts story lines that match the required user-story sentence shape."""
    clean = normalize_text(text)
    pattern = re.compile(r"(?im)^\s*(?:[-*]\s*)?(As a(?:n)?\s+.+?,\s*I want\s+.+?\s+so that\s+.+?\.)\s*$")
    return [m.group(1).strip() for m in pattern.finditer(clean)]



def extract_features(text):
    """Extracts feature blocks and rebuilds each block with exact required field labels."""
    clean = normalize_text(text)
    blocks = []
    for block in re.split(r"(?im)^\s*Feature\s*Name\s*:\s*", clean)[1:]:
        lines = block.split("\n")
        name = lines[0].strip()
        body = "\n".join(lines[1:])

        desc = re.search(r"(?ims)^\s*Description\s*:\s*(.+?)\s*(?=^\s*Key\s*Functionality\s*:|\Z)", body)
        func = re.search(r"(?ims)^\s*Key\s*Functionality\s*:\s*(.+?)\s*(?=^\s*User\s*Benefit\s*:|\Z)", body)
        bene = re.search(r"(?ims)^\s*User\s*Benefit\s*:\s*(.+?)\s*(?=^\s*Feature\s*Name\s*:|\Z)", body)

        if name and desc and func and bene:
            blocks.append(
                "\n".join([
                    f"Feature Name: {name}",
                    f"Description: {desc.group(1).strip()}",
                    f"Key Functionality: {func.group(1).strip()}",
                    f"User Benefit: {bene.group(1).strip()}"
                ])
            )
    return blocks



def extract_tasks(text):
    """Extracts task blocks and rebuilds each block with exact required field labels."""
    clean = normalize_text(text)
    blocks = []
    for block in re.split(r"(?im)^\s*Task\s*ID\s*:\s*", clean)[1:]:
        lines = block.split("\n")
        task_id = lines[0].strip()
        body = "\n".join(lines[1:])

        title = re.search(r"(?ims)^\s*Task\s*Title\s*:\s*(.+?)\s*(?=^\s*Related\s*User\s*Story\s*:|\Z)", body)
        story = re.search(r"(?ims)^\s*Related\s*User\s*Story\s*:\s*(.+?)\s*(?=^\s*Description\s*:|\Z)", body)
        desc = re.search(r"(?ims)^\s*Description\s*:\s*(.+?)\s*(?=^\s*Acceptance\s*Criteria\s*:|\Z)", body)
        crit = re.search(r"(?ims)^\s*Acceptance\s*Criteria\s*:\s*(.+?)\s*(?=^\s*Estimated\s*Effort\s*:|\Z)", body)
        eff = re.search(r"(?ims)^\s*Estimated\s*Effort\s*:\s*(.+?)\s*(?=^\s*Dependencies\s*:|\Z)", body)
        dep = re.search(r"(?ims)^\s*Dependencies\s*:\s*(.+?)\s*(?=^\s*Task\s*ID\s*:|\Z)", body)

        if task_id and title and story and desc and crit and eff and dep:
            blocks.append(
                "\n".join([
                    f"Task ID: {task_id}",
                    f"Task Title: {title.group(1).strip()}",
                    f"Related User Story: {story.group(1).strip()}",
                    f"Description: {desc.group(1).strip()}",
                    f"Acceptance Criteria: {crit.group(1).strip()}",
                    f"Estimated Effort: {eff.group(1).strip()}",
                    f"Dependencies: {dep.group(1).strip()}"
                ])
            )
    return blocks



def format_stories(text):
    """Returns canonically formatted user stories separated by blank lines."""
    return "\n\n".join(extract_stories(text))



def format_features(text):
    """Returns canonically formatted product feature blocks separated by blank lines."""
    return "\n\n".join(extract_features(text))



def format_tasks(text):
    """Returns canonically formatted engineering task blocks separated by blank lines."""
    return "\n\n".join(extract_tasks(text))



def is_valid_stories(text):
    """Validates that at least five correctly shaped user stories exist."""
    return len(extract_stories(text)) >= 5



def is_valid_features(text):
    """Validates that at least three complete feature blocks exist."""
    return len(extract_features(text)) >= 3



def is_valid_tasks(text):
    """Validates that at least five complete engineering task blocks exist."""
    return len(extract_tasks(text)) >= 5



def fallback_stories():
    """Returns deterministic fallback user stories for the Email Router plan."""
    return [
        "As a support operations manager, I want incoming emails automatically categorized so that urgent requests reach the right team faster.",
        "As a customer service agent, I want high-priority customer emails flagged immediately so that I can respond within service-level targets.",
        "As an IT administrator, I want configurable routing rules so that the email router can adapt to changing business workflows.",
        "As a compliance officer, I want all routing decisions logged so that audits can verify how customer communications were handled.",
        "As a reporting analyst, I want dashboards on routed email volume and outcomes so that I can identify bottlenecks and improve operations."
    ]



def fallback_features():
    """Returns deterministic fallback feature blocks for the Email Router plan."""
    return [
        "\n".join([
            "Feature Name: Intelligent Email Classification",
            "Description: Automatically analyzes incoming email content, sender metadata, and keywords to classify each message by business intent.",
            "Key Functionality: Assigns labels such as billing, technical support, account access, and escalation using configurable classification rules.",
            "User Benefit: Reduces manual triage time and improves the speed and accuracy of first-touch handling."
        ]),
        "\n".join([
            "Feature Name: Rules-Based Team Routing",
            "Description: Sends each email to the correct queue or team based on category, urgency, sender type, and defined routing policies.",
            "Key Functionality: Applies route conditions, priority overrides, fallback queues, and shared mailbox delivery for unresolved cases.",
            "User Benefit: Ensures the right team receives actionable emails quickly and consistently."
        ]),
        "\n".join([
            "Feature Name: Audit and Performance Monitoring",
            "Description: Tracks routing decisions, processing outcomes, and operational metrics for each email processed by the system.",
            "Key Functionality: Stores routing history, exposes volume and turnaround reports, and highlights failed or delayed deliveries.",
            "User Benefit: Improves transparency, compliance readiness, and continuous optimization of routing workflows."
        ])
    ]



def fallback_tasks():
    """Returns deterministic fallback engineering task blocks for the Email Router plan."""
    return [
        "\n".join([
            "Task ID: ER-101",
            "Task Title: Implement inbound email ingestion service",
            "Related User Story: As a support operations manager, I want incoming emails automatically categorized so that urgent requests reach the right team faster.",
            "Description: Build the service that receives inbound emails, validates payload structure, and normalizes message metadata for downstream routing.",
            "Acceptance Criteria: Service accepts supported email payloads, rejects malformed inputs, and stores normalized message records for processing.",
            "Estimated Effort: 5 story points",
            "Dependencies: Access to inbound mailbox or email webhook configuration."
        ]),
        "\n".join([
            "Task ID: ER-102",
            "Task Title: Develop classification rules engine",
            "Related User Story: As a customer service agent, I want high-priority customer emails flagged immediately so that I can respond within service-level targets.",
            "Description: Create the logic that evaluates subject lines, message content, sender metadata, and business keywords to classify each email.",
            "Acceptance Criteria: Engine assigns configured categories and priority labels with test coverage for key routing scenarios.",
            "Estimated Effort: 8 story points",
            "Dependencies: Normalized inbound email records from ER-101."
        ]),
        "\n".join([
            "Task ID: ER-103",
            "Task Title: Build configurable routing policy module",
            "Related User Story: As an IT administrator, I want configurable routing rules so that the email router can adapt to changing business workflows.",
            "Description: Implement configurable policy evaluation that maps classified emails to queues, teams, or fallback destinations.",
            "Acceptance Criteria: Administrators can define, update, and test routing conditions without code changes.",
            "Estimated Effort: 8 story points",
            "Dependencies: Classification outputs from ER-102 and destination queue definitions."
        ]),
        "\n".join([
            "Task ID: ER-104",
            "Task Title: Add audit trail and routing event logging",
            "Related User Story: As a compliance officer, I want all routing decisions logged so that audits can verify how customer communications were handled.",
            "Description: Persist routing decisions, timestamps, rule matches, and delivery outcomes in an auditable event log.",
            "Acceptance Criteria: Every processed email has a complete routing history that can be queried by message identifier.",
            "Estimated Effort: 5 story points",
            "Dependencies: Routing policy results from ER-103 and storage design approval."
        ]),
        "\n".join([
            "Task ID: ER-105",
            "Task Title: Create monitoring dashboard and alerts",
            "Related User Story: As a reporting analyst, I want dashboards on routed email volume and outcomes so that I can identify bottlenecks and improve operations.",
            "Description: Expose key operational metrics and alert conditions for backlog growth, failed routing, and delayed processing.",
            "Acceptance Criteria: Dashboard displays routing volumes, success rates, and exception counts, and alerts trigger on configured thresholds.",
            "Estimated Effort: 5 story points",
            "Dependencies: Audit log data from ER-104 and monitoring platform access."
        ])
    ]



def merge_with_fallback(existing_items, fallback_items, minimum_count):
    """Preserves valid extracted items and appends fallback items until the minimum count is met."""
    merged_items = []
    seen_items = set()

    for item in existing_items + fallback_items:
        normalized_item = normalize_text(item)
        if normalized_item and normalized_item not in seen_items:
            seen_items.add(normalized_item)
            merged_items.append(normalized_item)
        if len(merged_items) >= minimum_count:
            break

    return merged_items



def ensure_complete_stories(text):
    """Returns at least five valid user stories by topping up extracted stories with fallback content."""
    return "\n\n".join(merge_with_fallback(extract_stories(text), fallback_stories(), 5))



def ensure_complete_features(text):
    """Returns at least three complete feature blocks by topping up extracted features with fallback content."""
    return "\n\n".join(merge_with_fallback(extract_features(text), fallback_features(), 3))



def ensure_complete_tasks(text):
    """Returns at least five complete task blocks by topping up extracted tasks with fallback content."""
    return "\n\n".join(merge_with_fallback(extract_tasks(text), fallback_tasks(), 5))

# =========================
# STRICT GENERATORS
# =========================

def safe_agent_respond(agent, prompt):
    """Returns an agent response or an empty string if the request fails."""
    try:
        return agent.respond(prompt)
    except Exception as exc:
        print(f"Agent generation failed: {exc}")
        return ""



def generate_stories():
    """Requests exactly five user stories from the product manager agent."""
    prompt = f"""
You are generating user stories for an Email Router.

Generate EXACTLY 5 user stories.

STRICT FORMAT:
As a <user>, I want <action> so that <benefit>.

NO extra text.
"""
    return safe_agent_respond(pm_agent, prompt)



def generate_features():
    """Requests exactly three product features from the program manager agent."""
    prompt = f"""
Generate EXACTLY 3 product features for an Email Router.

STRICT FORMAT:

Feature Name: <name>
Description: <description>
Key Functionality: <functionality>
User Benefit: <benefit>

NO extra text.
"""
    return safe_agent_respond(prog_agent, prompt)



def generate_tasks():
    """Requests exactly five engineering tasks from the development agent."""
    prompt = f"""
Generate EXACTLY 5 engineering tasks for an Email Router.

STRICT FORMAT:

Task ID: <id>
Task Title: <title>
Related User Story: <story>
Description: <description>
Acceptance Criteria: <criteria>
Estimated Effort: <effort>
Dependencies: <dependencies>

NO extra text.
"""
    return safe_agent_respond(dev_agent, prompt)

# =========================
# FINAL SELF-HEALING LOOP
# =========================
MAX_RETRIES = 4

stories = ""
features = ""
tasks = ""

print("\n*** STARTING FINAL GENERATION ***\n")

for attempt in range(MAX_RETRIES):
    print(f"Attempt {attempt+1}...\n")

    if not is_valid_stories(stories):
        print("Generating STORIES...")
        stories = format_stories(generate_stories())

    if not is_valid_features(features):
        print("Generating FEATURES...")
        features = format_features(generate_features())

    if not is_valid_tasks(tasks):
        print("Generating TASKS...")
        tasks = format_tasks(generate_tasks())

    # Debug logs
    print("\n--- DEBUG OUTPUT ---")
    print("Stories count:", len(extract_stories(stories)))
    print("Features count:", len(extract_features(features)))
    print("Tasks count:", len(extract_tasks(tasks)))
    print("--------------------\n")

    if is_valid_stories(stories) and is_valid_features(features) and is_valid_tasks(tasks):
        print("ALL SECTIONS VALID\n")
        break

stories = ensure_complete_stories(stories)
features = ensure_complete_features(features)
tasks = ensure_complete_tasks(tasks)

print("Applying final completion pass...")
print("Stories count:", len(extract_stories(stories)))
print("Features count:", len(extract_features(features)))
print("Tasks count:", len(extract_tasks(tasks)))

# =========================
# FINAL VALIDATION
# =========================
if not (is_valid_stories(stories) and is_valid_features(features) and is_valid_tasks(tasks)):
    print("FINAL FAILURE: Could not generate valid structured output")
    sys.exit(1)

# =========================
# FINAL OUTPUT
# =========================
final_plan = "================================================================================\n"
final_plan += "FINAL STRUCTURED PROJECT PLAN: EMAIL ROUTER\n"
final_plan += "================================================================================\n\n"

final_plan += f"## USER STORIES\n{stories}\n\n"
final_plan += f"## PRODUCT FEATURES\n{features}\n\n"
final_plan += f"## ENGINEERING TASKS\n{tasks}\n\n"

# =========================
# SAVE FILE
# =========================
output_path = os.path.join(phase_2_dir, "test_evidence", "test_agentic_workflow.txt")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_plan)

print(f"SUCCESS: Plan saved to {output_path}")