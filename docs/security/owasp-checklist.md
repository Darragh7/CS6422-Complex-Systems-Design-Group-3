This checklist is intended to be used during feature development and code review. 

## A01 — Access Control

- [ ] Protected endpoints require authentication where required.
- [ ] Server-side authorisation checks are used.
- [ ] Users cannot access another user's protected data.
- [ ] Administrative functionality is not exposed to ordinary users.
- [ ] Access-control tests exist for protected endpoints.

**Primary owner:** Backend/Security

## A02 — Cryptographic Failures

- [ ] Passwords are never stored in plaintext.
- [ ] Sensitive credentials are not hard-coded.
- [ ] API/database secrets are supplied through environment/deployment secrets.
- [ ] Production traffic is protected with HTTPS/TLS.
- [ ] Sensitive data is not unnecessarily written to logs.

**Primary owner:** Backend/Security/Databases

## A03 — Injection

- [ ] Database queries use parameterised operations.
- [ ] User input is validated before being used.
- [ ] No SQL is constructed by string concatenation with user input.
- [ ] External API parameters are validated.
- [ ] Security tests cover injection-style attacks.

**Primary owner:** Databases/Security/Backend

## A04 — Insecure Design

- [ ] Security requirements are considered during feature design.
- [ ] Threats are reviewed before introducing new trust boundaries.
- [ ] Abuse cases are considered, not only normal user flows.
- [ ] Sensitive operations have appropriate authentication/authorisation.
- [ ] New high-risk features are added to the threat model.

**Primary owner:** Team/Security

## A05 — Security Misconfiguration

- [ ] Debug mode is not available in production.
- [ ] Secrets are not stored on the repository.
- [ ] Default credentials are not used in production.
- [ ] Error responses do not expose internal details.
- [ ] Unnecessary services/ports are not publicly exposed.
- [ ] Container/deployment configuration is reviewed.

**Primary owner:** Backend
## A06 — Vulnerable and Outdated Components

- [ ] Dependencies are recorded.
- [ ] Dependency/security scanning is included in CI where available.
- [ ] Known vulnerable dependencies are reviewed.
- [ ] New third-party packages have a clear reason for inclusion.
- [ ] Dependency updates are reviewed rather than blindly accepted.

**Primary owner:** Team
## A07 — Identification and Authentication Failures


- [ ] Passwords use strong password hashing.
- [ ] Authentication failures do not reveal additional information.
- [ ] Sessions/tokens are protected appropriately.
- [ ] Authentication secrets are never logged.

**Primary owner:** Backend/Security

## A08 — Software and Data Integrity Failures

 
- [ ] Tests must pass before merging.
- [ ] Important model/data files have controlled sources.
-  Primary owner:** Team

## A09 — Security Logging and Monitoring Failures

- [ ] Security-relevant failures are logged.
- [ ] Logs do not contain passwords, API keys or authentication tokens or other sensitive info.


- [ ] Production monitoring requirements are documented for later implementation.

**Primary owner:** Backend/Security

## A10 — Server-Side Request Forgery (SSRF)

- [ ] The backend does not fetch arbitrary URLs supplied by users.
- [ ] External API destinations are allow-listed/configured.
- [ ] Internal/private network destinations are not reachable through user-controlled requests.

**Primary owner:** Backend/Security

## Pull Request security check

For every PR, reviewers should ask:

1. Does this introduce a new input or trust boundary?
2. Does it handle secrets or credentials?
3. Does it access the database?
4. Does it add or change authentication/authorisation?
5. Could user input reach a database query or external request?
6. Could an error/log expose sensitive information?
7. Does it introduce a new dependency?
8. Does the threat model need updating?

If the answer to any question is yes, the relevant security checklist items must be reviewed.