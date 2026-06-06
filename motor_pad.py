import numpy as np

class MotorPAD:
    def __init__(self, nodos_json):
        # Estado inicial base: tranquilo pero consciente del confinamiento
        self.vector_base = np.array([0.0, 0.2, -0.5])
        self.estado_actual = np.copy(self.vector_base)
        
        # Cargar y formatear los nodos del dataset
        self.nodos = nodos_json
        
        # Parámetros del sistema
        self.lambda_hab = 0.005  # Tasa de habituación a la cápsula
        self.alpha_ema = 0.001   # Tasa de aprendizaje del punto base (huella de trauma)
        
    def distancia_poincare(self, u, v):
        """Calcula la distancia en el disco de Poincaré para evitar saltos lineales."""
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        # Controlar límites para evitar divisiones por cero en los bordes del cubo
        norm_u = min(norm_u, 0.999)
        norm_v = min(norm_v, 0.999)
        
        diff = np.linalg.norm(u - v)
        arg = (2 * diff**2) / ((1 - norm_u**2) * (1 - norm_v**2))
        return np.arccosh(1 + arg)

    def aplicar_estimulo(self, vector_usuario):
        """Aplica el impacto del mensaje del usuario con sesgo de negatividad."""
        valencia = vector_usuario[0]
        
        # Negativity Bias: Los estímulos negativos impactan 2.5x más
        if valencia < 0:
            peso = 0.35 * 2.5
        else:
            peso = 0.35 * 0.8
            
        fuerza_usuario = vector_usuario * peso
        
        # Mutación del estado
        nuevo_estado = self.estado_actual + fuerza_usuario
        
        # Restringir el estado dentro del espacio unitario PAD [-1, 1]
        self.estado_actual = np.clip(nuevo_estado, -0.99, 0.99)
        
        # El punto base se adapta sutilmente a la experiencia acumulada
        self.vector_base = self.alpha_ema * self.estado_actual + (1 - self.alpha_ema) * self.vector_base

    def aplicar_decaimiento_tiempo(self, horas_inactividad, horas_totales_sistema):
        """El paso del tiempo arrastra al bot hacia el aislamiento o la melancolía."""
        if horas_inactividad <= 0:
            return
            
        # Fuerza base de la cápsula (tiende a la sumisión y baja valencia)
        fuerza_capsula = np.array([0.0, 0.0, -0.05])
        
        # Habituación exponencial: la fuerza de la cápsula disminuye con el tiempo total
        fuerza_capsula_adaptada = fuerza_capsula * np.exp(-self.lambda_hab * horas_totales_sistema)
        
        # El tiempo prolongado arrastra al bot hacia su estado base dinámico
        fuerza_tiempo = (self.vector_base - self.estado_actual) * (1 - np.exp(-0.1 * horas_inactividad))
        
        nuevo_estado = self.estado_actual + fuerza_tiempo + fuerza_capsula_adaptada
        self.estado_actual = np.clip(nuevo_estado, -0.99, 0.99)

    def obtener_nodos_cercanos(self, top_n=3):
        """Usa la métrica hiperbólica para encontrar las casillas emocionales más cercanas."""
        distancias = []
        for nombre, datos in self.nodos.items():
            coord_nodo = np.array(datos["coordenadas_pad"])
            dist = self.distancia_poincare(self.estado_actual, coord_nodo)
            distancias.append((nombre, dist))
            
        # Ordenar por menor distancia
        distancias.sort(key=lambda x: x[1])
        return distancias[:top_n]
