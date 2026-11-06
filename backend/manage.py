import os
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in {'runserver', 'test', 'migrate', 'makemigrations', 'check', 'import_research', 'import_event_snapshot', 'createsuperuser'}:
        os.environ.setdefault('DJANGO_ENV', 'development')
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ipo_service.settings')
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
