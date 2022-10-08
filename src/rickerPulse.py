
import numpy as np
import math



class RickerPulse:

    def __init__(self, pulseConfig: list) -> None:
        """This class generates the Ricker's pulse object"""
        self.pulseIntensity     = pulseConfig[0]
        self.pulseFrequency     = pulseConfig[1]
        self.timeOfObservation  = pulseConfig[2]
        self.deltaTime          = pulseConfig[3]
        self.displacementFactor = pulseConfig[4]
        self.steps             = int(pulseConfig[2] /pulseConfig[3])+1
        self.time = np.zeros([1, self.steps],dtype=np.float32)
        self.pulse = np.zeros([1, self.steps],dtype=np.float32)

        for i in range(0, self.steps - 1):
            self.time[0, i + 1] = self.time[0, i] + self.deltaTime

        for i in range(0, self.steps):
            self.time[0, i] = self.time[0, i] - self.displacementFactor / self.pulseFrequency
            self.pulse[0, i] = self.pulseIntensity * ((1 - (2 * pow(math.pi, 2)) *
                                                       (pow(self.pulseFrequency, 2)) * (pow(self.time[0, i], 2))) *
                                                      pow(math.e,
                                                          ((-1) * (pow(math.pi, 2)) * (pow(self.pulseFrequency, 2))
                                                           * (pow(self.time[0, i], 2)))))
