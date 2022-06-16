import pybullet as p
import pybullet_data
import tensorflow as tf


def init_simulation(render = False):

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('optimized...')

    if render:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    p.setTimeStep(0.01)

    '------------------------------------'
    # drone
    drone = p.loadURDF('urdf/drone.urdf')

    # marker at desired point
    sphereVisualId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                        radius = 0.05,
                                        rgbaColor= [1, 0, 0, 1])
    marker = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=sphereVisualId, basePosition=[0, 0, 8.0],
                    useMaximalCoordinates=False)
    '-------------------------------------'
    p.resetDebugVisualizerCamera( cameraDistance=3.5, cameraYaw=30, 
                                cameraPitch=-10, cameraTargetPosition=[0,0,8])
                                
    return drone, marker

def end_simulation():
    p.disconnect()