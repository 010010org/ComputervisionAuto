
# ComputerVision Robot

Een open source project dat als doel heeft kinderen op een speelse wijze programmeervaardigheden bij te brengen. Dit project is gericht op het ontwerpen van een betaalbare robot, toegankelijk voor alle leeftijdsgroepen. Het streven is om de ontwerpen en code vrij beschikbaar te stellen, zodat anderen in staat zijn hun eigen versie van de robot te bouwen.

Voel je vrij om bij te dragen aan dit project door middel van feedback, suggesties, of zelfs door deel te nemen aan de ontwikkeling. Samen bouwen we aan een educatief hulpmiddel dat de wereld van programmeren toegankelijker maakt voor de jonge generatie.

Dit project wordt mogelijk gemaakt door Stichting 010010.
Dank voor je interesse en bijdrage. 


## Authors

- Prashant Chotkan


## Features

- Bewegingsfunctionaliteit:
    - Uitvoering van basisbewegingen, waaronder naar voren en naar achteren bewegen.
    - Draaien in graden: 90 naar links, 90 naar rechts, en 180.

- Programmeerblokken:
    - programmeerblokken met visuele representaties.
    - Blokken voor verschillende bewegingen, zoals vooruit en achteruit bewegen en draaien.

        ![img](https://raw.githubusercontent.com/010010org/ComputervisionAuto/main/blocks.png)

- Visuele Herkenning door AI:
    - Implementatie van kunstmatige intelligentie voor herkenning van visuele representaties op blokken.
    - Herkenning van specifieke commando's, zoals bewegen en draaien in verschillende graden.

- Besturingsbordje:
    - Besturingsbordje met ruimte voor 18 blokken met een gebruiksvriendelijke lay-out.
    ![img](https://raw.githubusercontent.com/010010org/ComputervisionAuto/main/besturingsbordje.jpg)
    - Mogelijkheid om het besturingsbordje uit een foto/video feed te halen en te isoleren door middel van edge detection en het filteren van contouren.
    ![img](https://raw.githubusercontent.com/010010org/ComputervisionAuto/main/edge_detection.png)

- Ontwerpen voor het bouwen van een mogelijke robot
![img](https://raw.githubusercontent.com/010010org/ComputervisionAuto/main/robot.jpg)
## Deployment
Volg de volgende stappen om het project werkend te krijgen op een Raspberry Pi 5

### Prerequisites
 - Zorg dat je een Raspberry Pi 5 hebt met [Raspberry Pi OS](https://www.raspberrypi.com/software/) geïnstalleerd
 - Zorg ervoor dat Python geïnstalleerd is op de Raspberry Pi
 - installeer alle benodigde Python bibliotheken met het commando:
```bash
pip install opencv-python pillow keras numpy picamera2 gpiozero
```
 - Installeer Tensorflow op de Raspberry Pi. Dit kan door compilatie wat langer duren. Volg de instructies op de [officiële Tensorflow website](https://www.tensorflow.org/install) om Tensorflow te installeren.

### Stappen om het programma te starten
 - clone deze repository naar de Raspberry Pi met het commando:
```bash
git clone https://github.com/010010org/ComputervisionAuto.git
cd ComputervisionAuto/final_program
```
 - start het Python script met het commando:
```bash
python converter.py
```

*voor het installeren van bibliotheken en het clonen van de repository is een internetverbinding nodig

### Model verder trainen
 - Voor het verbeteren en verder trainen van het AI model is een programma in map AI_training_program geplaatst. Om dit programma te gebruiken moeten de foto's uit je dataset in de juiste mappen geplaatst worden. Voor nieuwe klassen moet elk een nieuwe map aangemaakt worden in AI_training_program/training


## Contact

Prashant Chotkan - 1042569@hr.nl

