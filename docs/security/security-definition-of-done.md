# Security Definition of Done

The following security checks form part of the project's Definition of Done.

A feature should not be considered complete until the applicable checks below have been reviewed.

## Security DoD

- [ ] Input is validated at API boundaries.
- [ ] Authentication requirements have been considered.
- [ ] Authorisation requirements have been considered.
- [ ] No secrets, passwords, API keys or tokens are committed to Git.
- [ ] Database operations use safe parameterisation/ORM methods.
- [ ] Errors do not expose stack traces, credentials or other sensitive information.
- [ ] Logs do not contain passwords, API keys, tokens or other unnecessary sensitive data.
- [ ] Relevant OWASP risks have been considered.
- [ ] Security-sensitive behaviour has appropriate tests.
- [ ] New dependencies have been reviewed.
- [ ] CI/security checks pass.
- [ ] If the feature introduces a new threat or trust boundary, the threat model has been updated.
- [ ] Code review has been completed.

## Secrets-specific checks

- [ ] `.env` is not committed.
- [ ] `.env.example` contains placeholders only.
- [ ] No credentials are hard-coded in source code.
- [ ] Secret scanning passes.
- [ ] Kubernetes/deployment secrets are not stored as plaintext in the repository.

## Security exception

If an applicable security check cannot be completed for a feature, the exception must be documented in the pull request with the following details:

- what is incomplete;
- why it is incomplete;
- the risk created;
- who accepted the risk;
- the planned follow-up work.

Security exceptions should not be pushed without the teams knowledge.
