## 1. Purpose

This document identifies the main security risks for Scoreline and defines the initial mitigations that should be considered during development.

## 2. System overview

The planned architecture is a web-based sports prediction platform using:

- Frontend web application
- FastAPI backend
- SQL based database
- External football/fixtures API
- Prediction/ML model
- Docker containers
- Kubernetes for deployment

## 3. Assets to protect

| Asset | Why it matters | Security priority |
|---|---|---|
| API key | Prevents unauthorised use of the external API account | High |
| Database credentials | Could provide access to application data | High |
| Application secret key | May protect authentication/session-related functionality | High |
| User credentials | Prevents unauthorised users accessing sensitive info | High |
| Prediction data | Should not be modified or corrupted | Medium |
| ML/data files | Could affect prediction integrity | Medium |
| Application logs | Could expose internaland therefore sensitive information | Medium |
| Source code/repository | Could expose implementation details or secrets | Medium/High |

## 4. Trust boundaries

### Boundary 1 — User to frontend

The user controls input sent to the application.

**Threats:** malicious input, XSS attempts, unexpected values, abuse of endpoints.

### Boundary 2 — Frontend to FastAPI

Requests cross into the backend and must be treated as untrusted.

**Threats:** invalid input, unauthorised requests, injection attempts, excessive requests.

### Boundary 3 — FastAPI to PostgreSQL

The backend accesses persistent data.

**Threats:** SQL injection, leaked database credentials, excessive database permissions, data corruption.

### Boundary 4 — FastAPI to external football API

The application depends on an external service and API credential.

**Threats:** API key exposure, unexpected external responses, API failure, quota abuse.

### Boundary 5 — Backend to prediction/ML model

Application data is supplied to prediction logic.

**Threats:** malformed inputs, invalid model outputs, corrupted model/data files.

### Boundary 6 — Source repository / CI to deployment

Code and configuration move through GitHub/CI into containerised deployment.

**Threats:** committed secrets, vulnerable dependencies, compromised build artefacts, unsafe configuration.

##
## 5. Security controls

### Secrets

- Never hard-code API keys, passwords or application secrets.
- Local secrets should be supplied through environment variables.
- `.env` must not be committed.
- `.env.example` documents required variables without real credentials.
- Deployment secrets should be supplied through the deployment environment/Kubernetes Secrets.
- Secrets must not appear in application logs.

### Input validation

- Validate API request bodies, path parameters and query parameters.
- Use Pydantic schemas at FastAPI boundaries.
- Reject invalid ranges and unexpected values.
- Treat all client input as untrusted.

### Database security

- Use parameterised database operations.
- Do not construct SQL queries by concatenating user input.
- Use a dedicated application database account.
- Do not expose PostgreSQL directly to the public internet.

### Authentication and authorisation

- Protected functionality must require authentication.
- Authorisation must be checked server-side.
- Passwords must be stored using a strong password-hashing algorithm.
- Authentication tokens/secrets must not be logged.

### Error handling and logging

- Return consistent API errors.
- Do not expose stack traces, credentials, database connection strings or internal secrets to users.
- Logs should contain enough information for debugging without sensitive data.

### Dependencies and CI

- Run tests and linting in CI.
- Scan for accidentally committed secrets.
- Add dependency/security scanning as the project matures.
- Review new dependencies before introducing them.



