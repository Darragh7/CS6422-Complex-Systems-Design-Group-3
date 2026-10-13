# Secrets management

## Local development

1. Copy `.env.example` to `.env` and replace each placeholder with a local value.
2. Keep `.env` out of Git. The repository ignores `.env` and `.env.*`, while allowing the placeholder-only `.env.example`.
3. Settings are read from environment variables (and the local `.env` file). `DATABASE_URL` and `SECRET_KEY` have no insecure source-code defaults; configure them before starting the backend.
4. Do not paste credentials into issues, pull requests, chat, logs, screenshots, or test fixtures.

## Commit and CI scanning

- Install [Gitleaks](https://github.com/gitleaks/gitleaks) and run `gitleaks protect --staged --redact` before committing. This checks staged changes locally.
- GitHub Actions runs Gitleaks on pushes and pull requests. A finding fails the check; resolve the finding or rotate the exposed credential before merging. Do not add broad allowlists to hide real secrets.
- Local hooks are advisory and can be bypassed. CI is the required shared gate; repository administrators must require the secret scan check in branch protection for protected branches.

## If a secret is exposed

1. Revoke or rotate it immediately. Removing it from the latest commit does not invalidate a leaked credential or erase Git history.
2. Remove the secret from tracked files and history using the repository's incident process; coordinate history rewrites with collaborators.
3. Check provider audit logs for misuse and document the incident without copying the secret into the report.

## Deployment

Supply production values through the hosting platform's secret manager. Do not commit populated Kubernetes Secret manifests: base64 encoding is not encryption. The checked-in `kubernetes/secrets.yaml`, if used, must contain only a template or be generated from deployment secrets.
