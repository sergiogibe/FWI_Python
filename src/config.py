
from collections import namedtuple

environmentConfig = namedtuple('envConfig',
                               ['EL',
                                'ED',
                                'lenght',
                                'depth'
                                ])

materialConfig    = namedtuple('matConfig',
                               ['velocityList'
                                ])

pulseConfig       = namedtuple('pulConfig',
                               ['intensity',
                                'frequency',
                                'timeObservation',
                                'timeStep',
                                'displacementFactor'
                                ])

matmodConfig      = namedtuple('lsConfig' ,
                               ['deltaStep',
                                'numberIterations',
                                'diffusionCoefficient',
                                'materialModel'
                                ])

parametersConfig  = namedtuple('parConfig',
                               ['tolerance',
                                'normalizingFactor',
                                'regularizationFactor'
                                ])

sourcesDesign     = namedtuple('scDesign',
                               ['distanceFromOrigin',
                                'side1',
                                'side2',
                                'side3',
                                'side4'
                                ])

receiverDesign    = namedtuple('rcDesign',
                               ['distanceFromOrigin',
                                'side1',
                                'side2',
                                'side3',
                                'side4'
                                ])

# ID-1 ============================================================================================================

Id1Config = [environmentConfig(EL=100, ED=60, lenght=4.0, depth=2.5),

             materialConfig(velocityList=[3.5, 1.5]),

             pulseConfig(intensity=1, frequency=2.0, timeObservation=4.0, timeStep=0.002, displacementFactor=1.0),

             matmodConfig(deltaStep=0.03, numberIterations=100, diffusionCoefficient=5*pow(10,-6), materialModel="FWI"),

             parametersConfig(tolerance=pow(10,-3), normalizingFactor=0.1, regularizationFactor=0.0008)

             ]

Id1Design = [sourcesDesign(distanceFromOrigin=0, side1=1, side2=1, side3=2, side4=1),

             receiverDesign(distanceFromOrigin=0, side1=10, side2=10, side3=10, side4=10),

             ]

configList = [Id1Config]
designList = [Id1Design]












































#ignore
sens = 0
amm  = "Matmodel"
arr  = "(mesh,configList[config][3])"
cF   = []
cost0 = 0
misfit = 0
v = 0