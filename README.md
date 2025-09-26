# EmotionTracker

# Guía de Instalación de Dependencias para EmotionTracker

Este tutorial te explica cómo preparar tu entorno para ejecutar **EmotionTracker** en Linux.

---

## 1. Instalar `venv`

### En Linux

El módulo `venv` suele venir con Python 3, en caso de no tenerlo, puedes instalarlo como paquete en tu sistema:

```bash
sudo apt update
sudo apt install python3-venv
```

## 2. Crear un entorno virtual

### En Linux

```bash
python3 -m venv venv
```

## 3. Activar el entorno virtual

### En Linux

```bash
source venv/bin/activate
```

---

## 4. Instalar dependencias

Con el entorno virtual activado, instala las dependencias del proyecto:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Guía para instalar el modelo de IA mediante Ollama en un contenedor Docker

Esta guía te explica cómo instalar Docker en Debian 13 usando los paquetes `docker.io` y `docker-cli`, comprobar su funcionamiento, instalar Ollama en un contenedor Docker y descargar el modelo Qwen3:1.7b.

---

## 1. Instalar Docker

Abre una terminal y ejecuta:

```bash
sudo apt update
sudo apt install docker.io docker-cli
```

---

## 2. Habilitar y comprobar Docker

Activa el servicio de Docker y verifica que esté funcionando:

```bash
sudo systemctl enable --now docker
sudo systemctl status docker
```

Deberías ver que el servicio está "active (running)".

---

## 3. Instalar Ollama en Docker

Descarga y ejecuta el contenedor de Ollama:

```bash
sudo docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

- `-d`: Ejecuta el contenedor en segundo plano.
- `-v ollama:/root/.ollama`: Usa un volumen persistente para los modelos.
- `-p 11434:11434`: Expone el puerto 11434.
- `--name ollama`: Nombra el contenedor como "ollama".

Verifica que el contenedor esté corriendo:

```bash
sudo docker ps
```

---

## 4. Instalar el modelo Qwen3:1.7b en Ollama

Con el contenedor Ollama en ejecución, descarga el modelo Qwen3:1.7b:

```bash
sudo docker exec -it ollama ollama run qwen3:1.7b
```

Esto descargará e iniciará el modelo por primera vez (puede tardar varios minutos).

---

## 5. Verificar el funcionamiento de Ollama

Para comprobar que Ollama y el modelo funcionan correctamente, puedes ejecutar:

```bash
curl http://localhost:11434/api/tags
```

Deberías ver una lista de los modelos instalados, incluyendo `qwen3:1.7b`.

---

¡Listo! Ahora tienes Docker y Ollama funcionando con el modelo Qwen3:1.7b en Debian 13.