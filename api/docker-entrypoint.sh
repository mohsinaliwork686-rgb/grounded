#!/bin/sh
# Index the documents in $GROUNDED_DOCS, then start the server. Unchanged files are skipped,
# so restarts are quick. A file that fails to parse is reported but doesn't stop the server.
set -u

if [ -d "${GROUNDED_DOCS}" ]; then
    grounded ingest "${GROUNDED_DOCS}"
    status=$?
    if [ "$status" -eq 2 ]; then
        echo "grounded: ingest failed, not starting the server" >&2
        exit 2
    fi
else
    echo "grounded: no documents at ${GROUNDED_DOCS}; mount a folder there" >&2
fi

exec "$@"
