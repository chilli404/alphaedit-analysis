#!/usr/bin/env bash
# Patch baselines/EvoEdit and baselines/REVIVEEDIT for compatibility.
# Called from the YAML run block or manually on clusters.
# Idempotent — safe to run multiple times.
set -euo pipefail

PROJECT_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
EVOEDIT="$PROJECT_DIR/baselines/EvoEdit"

echo "=== Patching baselines for compatibility ==="

# Fix EvoEdit apply function to accept **kwargs
_EVOEDIT_MAIN="$EVOEDIT/EvoEdit/EvoEdit_main.py"
if [ -f "$_EVOEDIT_MAIN" ]; then
    # Write the correct function signature directly
    python3 -c "
import re
t = open('$_EVOEDIT_MAIN').read()
# Match the entire apply_EvoEdit_to_model signature and normalize it
old = re.search(r'def apply_EvoEdit_to_model\([^)]*\)', t, re.DOTALL)
if old:
    sig = old.group()
    # Always normalize: strip _kwargs from all lines, then add exactly one
    lines = sig.split('\n')
    clean = []
    for l in lines:
        stripped = l.replace(', **_kwargs,', ',').replace(', **_kwargs', '').replace('**_kwargs,', '').replace('**_kwargs', '')
        if stripped.strip() and stripped.strip() != ',':
            clean.append(stripped)
    last = clean[-1]
    if last.strip() == ')':
        clean.insert(-1, '    **_kwargs')
    else:
        clean.append('    **_kwargs')
        clean.append(')')
    for i in range(len(clean)-1):
        if clean[i+1].strip().startswith('**_kwargs'):
            cl = clean[i].rstrip()
            if not cl.endswith(','):
                clean[i] = cl + ','
            break
    new_sig = '\n'.join(clean)
    t = t[:old.start()] + new_sig + t[old.end():]
    open('$_EVOEDIT_MAIN','w').write(t)
    print('  EvoEdit: normalized **_kwargs' if new_sig != sig else '  EvoEdit: verified correct')
"
fi

# Fix NSE apply function to accept **kwargs
_NSE_MAIN="$EVOEDIT/nse/nse_main.py"
if [ -f "$_NSE_MAIN" ]; then
    python3 -c "
import re
t = open('$_NSE_MAIN').read()
old = re.search(r'def apply_nse_to_model\([^)]*\)', t, re.DOTALL)
if old:
    sig = old.group()
    lines = sig.split('\n')
    clean = []
    for l in lines:
        stripped = l.replace(', **_kwargs,', ',').replace(', **_kwargs', '').replace('**_kwargs,', '').replace('**_kwargs', '')
        if stripped.strip() and stripped.strip() != ',':
            clean.append(stripped)
    last = clean[-1]
    if last.strip() == ')':
        clean.insert(-1, '    **_kwargs')
    else:
        clean.append('    **_kwargs')
        clean.append(')')
    for i in range(len(clean)-1):
        if clean[i+1].strip().startswith('**_kwargs'):
            cl = clean[i].rstrip()
            if not cl.endswith(','):
                clean[i] = cl + ','
            break
    new_sig = '\n'.join(clean)
    t = t[:old.start()] + new_sig + t[old.end():]
    open('$_NSE_MAIN','w').write(t)
    print('  NSE: normalized **_kwargs' if new_sig != sig else '  NSE: verified correct')
"
fi

# Fix MEMIT_rect apply function to accept **kwargs
_RECT_MAIN="$EVOEDIT/memit/memit_rect_main.py"
if [ -f "$_RECT_MAIN" ]; then
    python3 -c "
import re
t = open('$_RECT_MAIN').read()
old = re.search(r'def apply_memit_rect_to_model\([^)]*\)', t, re.DOTALL)
if old:
    sig = old.group()
    if '**_kwargs' not in sig:
        lines = sig.split('\n')
        last = lines[-1]
        if last.strip() == ')':
            lines.insert(-1, '    **_kwargs,')
        else:
            lines[-1] = last.rstrip().rstrip(')').rstrip(',') + ','
            lines.append('    **_kwargs,')
            lines.append(')')
        new_sig = '\n'.join(lines)
        t = t[:old.start()] + new_sig + t[old.end():]
        open('$_RECT_MAIN','w').write(t)
        print('  MEMIT_rect: added **_kwargs')
    else:
        print('  MEMIT_rect: verified correct')
"
fi

# Fix MEMIT_seq apply function to accept **kwargs
_SEQ_MAIN="$EVOEDIT/memit/memit_seq_main.py"
if [ -f "$_SEQ_MAIN" ]; then
    python3 -c "
import re
t = open('$_SEQ_MAIN').read()
old = re.search(r'def apply_memit_seq_to_model\([^)]*\)', t, re.DOTALL)
if old:
    sig = old.group()
    if '**_kwargs' not in sig:
        lines = sig.split('\n')
        last = lines[-1]
        if last.strip() == ')':
            lines.insert(-1, '    **_kwargs,')
        else:
            lines[-1] = last.rstrip().rstrip(')').rstrip(',') + ','
            lines.append('    **_kwargs,')
            lines.append(')')
        new_sig = '\n'.join(lines)
        t = t[:old.start()] + new_sig + t[old.end():]
        open('$_SEQ_MAIN','w').write(t)
        print('  MEMIT_seq: added **_kwargs')
    else:
        print('  MEMIT_seq: verified correct')
"
fi

# Fix MEMIT_seq_rect apply function to accept **kwargs
_SEQ_RECT_MAIN="$EVOEDIT/memit/memit_seq_rect_main.py"
if [ -f "$_SEQ_RECT_MAIN" ]; then
    python3 -c "
import re
t = open('$_SEQ_RECT_MAIN').read()
old = re.search(r'def apply_memit_seq_rect_to_model\([^)]*\)', t, re.DOTALL)
if old:
    sig = old.group()
    if '**_kwargs' not in sig:
        lines = sig.split('\n')
        last = lines[-1]
        if last.strip() == ')':
            lines.insert(-1, '    **_kwargs,')
        else:
            lines[-1] = last.rstrip().rstrip(')').rstrip(',') + ','
            lines.append('    **_kwargs,')
            lines.append(')')
        new_sig = '\n'.join(lines)
        t = t[:old.start()] + new_sig + t[old.end():]
        open('$_SEQ_RECT_MAIN','w').write(t)
        print('  MEMIT_seq_rect: added **_kwargs')
    else:
        print('  MEMIT_seq_rect: verified correct')
"
fi

# ============================================================
# Also patch vendor/AlphaEdit/ — polykernel_seqreg_runner imports from here
# ============================================================
VENDOR="$PROJECT_DIR/vendor/AlphaEdit"

# Vendor MEMIT
_V_MEMIT="$VENDOR/memit/memit_main.py"
if [ -f "$_V_MEMIT" ]; then
    grep -q '_kwargs' "$_V_MEMIT" || \
        sed -i.bak 's/    cache_template: Optional\[str\] = None,/    cache_template: Optional[str] = None, **_kwargs,/' "$_V_MEMIT"
fi

# Vendor AlphaEdit
_V_AE="$VENDOR/AlphaEdit/AlphaEdit_main.py"
if [ -f "$_V_AE" ]; then
    grep -q '_kwargs' "$_V_AE" || \
        sed -i.bak 's/    P = None,/    P = None, **_kwargs,/' "$_V_AE"
fi

# Vendor NSE
_V_NSE="$VENDOR/nse/nse_main.py"
if [ -f "$_V_NSE" ]; then
    grep -q '_kwargs' "$_V_NSE" || \
        sed -i.bak 's/    cache_c = None,/    cache_c = None, **_kwargs,/' "$_V_NSE"
fi

# Vendor MEMIT_rect
_V_RECT="$VENDOR/memit/memit_rect_main.py"
if [ -f "$_V_RECT" ]; then
    grep -q '_kwargs' "$_V_RECT" || \
        sed -i.bak 's/    cache_template: Optional\[str\] = None,/    cache_template: Optional[str] = None, **_kwargs,/' "$_V_RECT"
fi

echo "=== Baselines + vendor patched ==="
