import ast, sys
p='src/scheduler/runner.py'
with open(p,'r',encoding='utf-8') as f:
    src=f.read()
try:
    t=ast.parse(src,p)
    print('OK')
except SyntaxError as e:
    print('SyntaxError', e.lineno, e.offset, e.text.strip() if e.text else '')
