FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

FROM nginx:1.27-alpine
RUN apk add --no-cache openssl
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
# Runs before nginx starts: login required when BASIC_AUTH_USER/PASSWORD are set.
COPY --chmod=755 <<'EOF' /docker-entrypoint.d/40-basic-auth.sh
#!/bin/sh
set -eu
mkdir -p /etc/nginx/snippets
if [ -n "${BASIC_AUTH_USER:-}" ] && [ -n "${BASIC_AUTH_PASSWORD:-}" ]; then
  printf '%s:%s\n' "$BASIC_AUTH_USER" "$(openssl passwd -apr1 "$BASIC_AUTH_PASSWORD")" > /etc/nginx/.htpasswd
  chmod 640 /etc/nginx/.htpasswd
  chgrp nginx /etc/nginx/.htpasswd 2>/dev/null || true
  printf 'auth_basic "ATLAS";\nauth_basic_user_file /etc/nginx/.htpasswd;\n' > /etc/nginx/snippets/auth.conf
  echo "40-basic-auth: login required for user '$BASIC_AUTH_USER'"
else
  : > /etc/nginx/snippets/auth.conf
  echo "40-basic-auth: BASIC_AUTH_USER/PASSWORD not set; site is open (local use only)"
fi
EOF
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
