# pln_p3_7461_07

Práctica 3 de PLN. El código está en GitHub y los datos se versionan con **DVC** y se guardan en **Google Drive**.

**uv** (gestor de Python y dependencias)

curl -LsSf https://astral.sh/uv/install.sh | sh (linux)

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" (Windows)

### 2. Clonar el repositorio

git clone https://github.com/idom-ai/pln_p3_7461_07.git
cd pln_p3_7461_07

### 3. Instalar las dependencias

uv sync

Este comando crea el entorno virtual `.venv` e instala la versión de Python del proyecto, DVC, `dvc-gdrive` y el resto de librerías.

Comprueba que DVC funciona y es compatible con Google Drive:

uv run dvc version

En la línea `Supports:` debe aparecer `gdrive`.

Si usas conda, desactívalo antes (`conda deactivate`) para no mezclar entornos.

### 4. Descargar los datos

uv run dvc pull

### Bajar los últimos cambios

git pull
uv run dvc pull

### Subir cambios solo de código

git add .
git commit -m "Descripción del cambio"
git push

### Subir cambios en los datos

uv run dvc add <ruta/a/los/datos>
git add <ruta/a/los/datos>.dvc .gitignore
git commit -m "Actualizar datos"
uv run dvc push
git push