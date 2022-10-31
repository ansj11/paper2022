#!/usr/bin/env bash

_lsk_script_path="${BASH_SOURCE[0]}"
landstack=$(dirname "$(readlink -f "$_lsk_script_path")")
script=$(basename "$_lsk_script_path" | sed -e 's/^lsk-//g')
script_file="$landstack/scripts/${script}.py"

exec "mdl" "${script_file}" "$@"
