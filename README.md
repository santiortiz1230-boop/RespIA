# RespIA

Prototipo académico de investigación (Universidad Manuela Beltrán) que clasifica el patrón acústico de un ciclo respiratorio en normal, crepitante o sibilancia. No es un dispositivo médico y no sirve para diagnosticar.

La aplicación recibe un archivo WAV con una sola respiración, muestra su forma de onda y su espectrograma, y entrega la clase con la probabilidad de cada una.

Para ejecutarla en un computador:

```bash
python -m pip install -r requirements.txt
```

```bash
streamlit run app/aplicacion.py
```

Los seis audios de ejemplo son ciclos del dataset ICBHI 2017 (Rocha et al.) y se incluyen solo con fines académicos.
