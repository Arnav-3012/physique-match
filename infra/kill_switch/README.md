# Kill switch (ADR-028, ADR-031)

Pub/Sub topic `budget-alerts` (project `physique-gap-3012`) -> this function -> unlink billing when cost >= budget.

Ships with `SIMULATE_DEACTIVATION=true`. The real trip test (`false`) is a separate, user-approved step in P7.

## Deploy (simulation mode)
Needs `cloudfunctions`, `cloudbuild`, `run`, `artifactregistry`, `eventarc` APIs enabled, and the function's service account needs
`roles/billing.projectManager` on the project (and billing admin on the billing account) before a real trip works.

```
gcloud functions deploy kill-switch \
  --gen2 --runtime=python312 --region=<REGION> \
  --source=infra/kill_switch --entry-point=stop_billing \
  --trigger-topic=budget-alerts \
  --set-env-vars=GCP_PROJECT=physique-gap-3012,SIMULATE_DEACTIVATION=true \
  --max-instances=1
```

## Test with a synthetic over-budget message
```
gcloud pubsub topics publish budget-alerts \
  --message='{"costAmount": 150, "budgetAmount": 100, "budgetDisplayName": "synthetic"}'
gcloud functions logs read kill-switch --gen2 --region=<REGION> --limit=20
```
Expected: `Billing disabled. (Simulated)`, and `gcloud billing projects describe physique-gap-3012` still shows `billingEnabled: true`.
Also publish `costAmount: 10` and expect `Under budget, nothing to do.`
The function reads the project's billing info even in simulation and does not catch errors: a permission error in the log means the service account still needs a billing role (see above), not that the code is broken.

## Budget note
Create the budget with credits **not** subtracted (gross cost), otherwise trial credits hide spend and the switch never fires.
