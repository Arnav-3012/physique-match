"""Budget kill switch (ADR-028): detach billing from the project once cost reaches the budget.

Triggered by the budget's Pub/Sub topic. Detaching billing stops EVERY service in the project,
including free-tier ones, and resources may be lost. Cost notifications lag by hours, so the cap
is not exact. Based on Google's "disable billing usage with notifications" example.

SIMULATE_DEACTIVATION defaults to true: the function logs what it would do and detaches nothing.
Set it to "false" only after a simulated run worked and the user approved a real trip test.
"""

import base64
import json
import os

import functions_framework
from google.cloud import billing_v1

PROJECT_NAME = f"projects/{os.environ['GCP_PROJECT']}"
SIMULATE = os.environ.get("SIMULATE_DEACTIVATION", "true").lower() != "false"


@functions_framework.cloud_event
def stop_billing(cloud_event):
    msg = json.loads(base64.b64decode(cloud_event.data["message"]["data"]))
    cost, budget = msg["costAmount"], msg["budgetAmount"]
    print(f"budget={msg.get('budgetDisplayName')} cost={cost} budget_amount={budget}")
    if cost < budget:
        print("Under budget, nothing to do.")
        return

    client = billing_v1.CloudBillingClient()
    # Not caught on purpose: a missing permission must fail loudly, in simulation too.
    if not client.get_project_billing_info(name=PROJECT_NAME).billing_enabled:
        print("Billing already disabled.")
        return
    if SIMULATE:
        print("Billing disabled. (Simulated)")
        return
    client.update_project_billing_info(
        name=PROJECT_NAME,
        project_billing_info=billing_v1.ProjectBillingInfo(billing_account_name=""),
    )
    print("Billing disabled. All services in the project are stopping.")
