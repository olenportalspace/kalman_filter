from numpy import angle
from rocketpy import Environment, SolidMotor, Rocket, Flight
import matplotlib.pyplot as plt
import datetime
import numpy as np
# %matplotlib widget

Env = Environment(
    railLength=5.2, latitude=32.990254, longitude=-106.974998, elevation=120
)


tomorrow = datetime.date.today() + datetime.timedelta(days=1)

Env.setDate((tomorrow.year, tomorrow.month, tomorrow.day, 12))  # Hour given in UTC time

# Env.setAtmosphericModel(type="Forecast", file="GFS")

Pro75M1670 = SolidMotor(
    thrustSource="./data/motors/Cesaroni_M1670.eng",
    burnOut=3.9,
    grainNumber=5,
    grainSeparation=5 / 1000,
    grainDensity=1815,
    grainOuterRadius=33 / 1000,
    grainInitialInnerRadius=15 / 1000,
    grainInitialHeight=120 / 1000,
    nozzleRadius=33 / 1000,
    throatRadius=11 / 1000,
    interpolationMethod="linear"
)


Calisto = Rocket(
    motor=Pro75M1670,
    radius=127 / 2000,
    mass=19.197 - 2.956,
    inertiaI=6.60,
    inertiaZ=0.0351,
    distanceRocketNozzle=-1.255,
    distanceRocketPropellant=-0.85704,
    powerOffDrag="./data/calisto/powerOffDragCurve.csv",
    powerOnDrag="./data/calisto/powerOnDragCurve.csv",
)

Calisto.setRailButtons([0.2, -0.5])


NoseCone = Calisto.addNose(length=0.55829, kind="vonKarman", distanceToCM=0.71971)

FinSet = Calisto.addFins(
    4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956
)

Tail = Calisto.addTail(
    topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656
)

def drogueTrigger(p, y):
    # p = pressure
    # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
    # activate drogue when vz < 0 m/s.
    return True if y[5] < 0 else False


def mainTrigger(p, y):
    # p = pressure
    # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
    # activate main when vz < 0 m/s and z < 800 + 1400 m (+1400 due to surface elevation).
    return True if y[5] < 0 and y[2] < 800 + 1400 else False


Main = Calisto.addParachute(
    "Main",
    CdS=10.0,
    trigger=mainTrigger,
    samplingRate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)

Drogue = Calisto.addParachute(
    "Drogue",
    CdS=1.0,
    trigger=drogueTrigger,
    samplingRate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)


TestFlight = Flight(rocket=Calisto, environment=Env, inclination=90, heading=0)

TestFlight.exportKML(
    fileName="trajectory.kml",
    extrude=True,
    altitudeMode="relativetoground",
)

# TestFlight.


fig, ax = plt.subplots()
ax.plot([x[0] for x in TestFlight.az], [x[1] for x in TestFlight.az], linewidth=2.0, label="az")
plt.title("Up")
plt.show()


fig, ax = plt.subplots()
ax.plot([x[0] for x in TestFlight.ay], [x[1] for x in TestFlight.ay], linewidth=2.0, label="ay")
plt.title("y")
plt.show()

fig, ax = plt.subplots()
ax.plot([x[0] for x in TestFlight.ax], [x[1] for x in TestFlight.ax], linewidth=2.0, label="ax")
plt.title("x")
plt.show()

fig, ax = plt.subplots()
ax.plot([x[0] for x in TestFlight.ax], [x[1] for x in TestFlight.ax], linewidth=2.0, label="ax")
plt.title("x")
plt.show()

thaspeed = np.asarray(TestFlight.speed)


# print thaspeed as graph where first column is time and second is speed
fig, ax = plt.subplots()
ax.plot([x[0] for x in thaspeed], [x[1] for x in thaspeed], linewidth=2.0, label="speed")
plt.title("speed")
plt.show()




# ax.plot([x[0] for x in TestFlight.attitudeVectorY], [x[1] for x in TestFlight.attitudeVectorY], linewidth=2.0)
# ax.plot([x[0] for x in TestFlight.attitudeVectorZ], [x[1] for x in TestFlight.attitudeVectorZ], linewidth=2.0)

plt.show()