#!/usr/bin/env python3
"""
Script для автоматического рефакторинга admin command файлов
Удаляет дублирующиеся helper функции и заменяет их импортами из admin_formatters
"""

import re
from pathlib import Path

# Mapping old functions to new imports
FORMATTERS_MAP = {
    '_format_trend': 'format_trend',
    '_nps_emoji': 'nps_emoji',
    '_growth_emoji': 'growth_emoji',
    '_stickiness_emoji': 'stickiness_emoji',
    '_quick_ratio_status': 'quick_ratio_status',
    '_ltv_cac_status': 'ltv_cac_status',
    '_pmf_status': 'pmf_status',
    '_pmf_rating_emoji': 'pmf_rating_emoji',
}

def refactor_file(filepath: Path):
    """Refactor single file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Collect which formatters are used in this file
    used_formatters = []
    for old_name, new_name in FORMATTERS_MAP.items():
        if old_name + '(' in content:
            used_formatters.append(new_name)

    if not used_formatters:
        print(f"  ✓ {filepath.name} - no formatters to refactor")
        return

    # Add import if not present
    import_line = f"from src.core.admin_modules.admin_formatters import {', '.join(used_formatters)}"

    if 'from src.core.admin_modules.admin_formatters import' not in content:
        # Find where to insert import (after other imports)
        import_section_match = re.search(r'(from src\.core\.admin_modules\.[^\n]+\n)', content)
        if import_section_match:
            last_import_pos = import_section_match.end()
            content = content[:last_import_pos] + import_line + '\n' + content[last_import_pos:]

    # Remove local function definitions
    for old_name in FORMATTERS_MAP.keys():
        # Pattern to match function definition and body
        pattern = rf'def {old_name}\([^)]+\)[^:]*:.*?(?=\n(?:def |async def |@|\Z))'
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    # Replace function calls
    for old_name, new_name in FORMATTERS_MAP.items():
        content = content.replace(f'{old_name}(', f'{new_name}(')

    # Clean up extra blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ {filepath.name} - refactored ({len(used_formatters)} formatters)")
    else:
        print(f"  ✓ {filepath.name} - no changes")

def main():
    admin_modules_dir = Path('src/core/admin_modules')

    files_to_refactor = [
        'admin_pmf_commands.py',
        'admin_revenue_commands.py',
    ]

    print("🔧 Refactoring admin command files...\n")

    for filename in files_to_refactor:
        filepath = admin_modules_dir / filename
        if filepath.exists():
            refactor_file(filepath)
        else:
            print(f"  ✗ {filename} - file not found")

    print("\n✅ Refactoring complete!")

if __name__ == '__main__':
    main()
