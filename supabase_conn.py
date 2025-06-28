# supabase_conn.py
import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Obtener credenciales
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Crear cliente global
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)