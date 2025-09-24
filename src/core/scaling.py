"""
Компоненты для horizontal scaling и load balancing
"""

from __future__ import annotations
import os
import json
import asyncio
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
import uuid

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class NodeInfo:
    """Информация о ноде в кластере"""
    node_id: str
    host: str
    port: int
    started_at: datetime
    last_heartbeat: datetime
    status: str = "active"  # active, inactive, maintenance
    load_score: float = 0.0  # 0.0 - минимальная нагрузка, 1.0 - максимальная
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        """Проверка здоровья ноды"""
        if self.status != "active":
            return False
        
        # Нода считается нездоровой если не было heartbeat более 30 секунд
        heartbeat_timeout = timedelta(seconds=30)
        return (datetime.now() - self.last_heartbeat) < heartbeat_timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация для Redis"""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "started_at": self.started_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "status": self.status,
            "load_score": self.load_score,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeInfo':
        """Десериализация из Redis"""
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            started_at=datetime.fromisoformat(data["started_at"]),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            status=data.get("status", "active"),
            load_score=data.get("load_score", 0.0),
            metadata=data.get("metadata", {})
        )

class ServiceRegistry:
    """Реестр сервисов для service discovery"""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        node_id: Optional[str] = None,
        heartbeat_interval: float = 15.0,
        cleanup_interval: float = 60.0
    ):
        self.redis_url = redis_url
        self.node_id = node_id or self._generate_node_id()
        self.heartbeat_interval = heartbeat_interval
        self.cleanup_interval = cleanup_interval
        
        # Redis подключение
        self._redis: Optional[redis.Redis] = None
        
        # Локальный fallback
        self._local_nodes: Dict[str, NodeInfo] = {}
        
        # Текущая нода
        self.current_node: Optional[NodeInfo] = None
        
        # Фоновые задачи
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    def _generate_node_id(self) -> str:
        """Генерация уникального ID ноды"""
        hostname = os.getenv("HOSTNAME", "localhost")
        pid = os.getpid()
        timestamp = datetime.now().isoformat()
        
        unique_string = f"{hostname}:{pid}:{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    async def initialize(self) -> None:
        """Инициализация реестра"""
        # Подключение к Redis если доступен
        if self.redis_url and REDIS_AVAILABLE:
            try:
                self._redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self._redis.ping()
                logger.info("Connected to Redis service registry")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis registry: {e}")
                self._redis = None
        
        # Создаем информацию о текущей ноде
        self.current_node = NodeInfo(
            node_id=self.node_id,
            host=os.getenv("HOST", "localhost"),
            port=int(os.getenv("PORT", "8000")),
            started_at=datetime.now(),
            last_heartbeat=datetime.now()
        )
        
        # Регистрируем ноду
        await self.register_node(self.current_node)
    
    async def register_node(self, node: NodeInfo) -> None:
        """Регистрация ноды в реестре"""
        if self._redis:
            try:
                await self._redis.hset(
                    "nodes",
                    node.node_id,
                    json.dumps(node.to_dict())
                )
                logger.info(f"Registered node {node.node_id} in Redis registry")
            except Exception as e:
                logger.error(f"Failed to register node in Redis: {e}")
        
        # Локальный fallback
        self._local_nodes[node.node_id] = node
    
    async def unregister_node(self, node_id: str) -> None:
        """Удаление ноды из реестра"""
        if self._redis:
            try:
                await self._redis.hdel("nodes", node_id)
                logger.info(f"Unregistered node {node_id} from Redis registry")
            except Exception as e:
                logger.error(f"Failed to unregister node from Redis: {e}")
        
        # Локальный fallback
        self._local_nodes.pop(node_id, None)
    
    async def get_active_nodes(self) -> List[NodeInfo]:
        """Получение списка активных нод"""
        nodes = []
        
        if self._redis:
            try:
                nodes_data = await self._redis.hgetall("nodes")
                for node_id, node_json in nodes_data.items():
                    try:
                        node_dict = json.loads(node_json)
                        node = NodeInfo.from_dict(node_dict)
                        if node.is_healthy:
                            nodes.append(node)
                    except Exception as e:
                        logger.warning(f"Failed to parse node data for {node_id}: {e}")
            except Exception as e:
                logger.error(f"Failed to get nodes from Redis: {e}")
        
        # Fallback на локальные ноды
        if not nodes:
            nodes = [node for node in self._local_nodes.values() if node.is_healthy]
        
        return nodes
    
    async def update_node_load(self, node_id: str, load_score: float) -> None:
        """Обновление информации о нагрузке ноды"""
        if self._redis:
            try:
                node_json = await self._redis.hget("nodes", node_id)
                if node_json:
                    node_dict = json.loads(node_json)
                    node_dict["load_score"] = load_score
                    node_dict["last_heartbeat"] = datetime.now().isoformat()
                    
                    await self._redis.hset(
                        "nodes",
                        node_id,
                        json.dumps(node_dict)
                    )
            except Exception as e:
                logger.error(f"Failed to update node load: {e}")
        
        # Локальное обновление
        if node_id in self._local_nodes:
            self._local_nodes[node_id].load_score = load_score
            self._local_nodes[node_id].last_heartbeat = datetime.now()
    
    async def start_background_tasks(self) -> None:
        """Запуск фоновых задач"""
        if self._running:
            return
        
        self._running = True
        
        # Heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Started service registry background tasks")
    
    async def stop_background_tasks(self) -> None:
        """Остановка фоновых задач"""
        self._running = False
        
        # Отменяем задачи
        for task in [self._heartbeat_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Снимаем ноду с регистрации
        if self.current_node:
            await self.unregister_node(self.current_node.node_id)
        
        # Закрываем Redis подключение
        if self._redis:
            await self._redis.close()
        
        logger.info("Stopped service registry background tasks")
    
    async def _heartbeat_loop(self) -> None:
        """Цикл отправки heartbeat"""
        try:
            while self._running:
                if self.current_node:
                    self.current_node.last_heartbeat = datetime.now()
                    await self.register_node(self.current_node)
                
                await asyncio.sleep(self.heartbeat_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat loop failed: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Цикл очистки мертвых нод"""
        try:
            while self._running:
                await asyncio.sleep(self.cleanup_interval)
                
                if self._redis:
                    try:
                        nodes_data = await self._redis.hgetall("nodes")
                        dead_nodes = []
                        
                        for node_id, node_json in nodes_data.items():
                            try:
                                node_dict = json.loads(node_json)
                                node = NodeInfo.from_dict(node_dict)
                                if not node.is_healthy:
                                    dead_nodes.append(node_id)
                            except:
                                dead_nodes.append(node_id)  # Некорректные данные тоже удаляем
                        
                        if dead_nodes:
                            await self._redis.hdel("nodes", *dead_nodes)
                            logger.info(f"Cleaned up {len(dead_nodes)} dead nodes")
                            
                    except Exception as e:
                        logger.error(f"Cleanup task failed: {e}")
                
                # Локальная очистка
                dead_local = [
                    node_id for node_id, node in self._local_nodes.items()
                    if not node.is_healthy
                ]
                for node_id in dead_local:
                    del self._local_nodes[node_id]
        
        except asyncio.CancelledError:
            pass

class LoadBalancer:
    """Load balancer с различными стратегиями распределения"""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.request_count = 0
        
        # Статистика
        self.total_requests = 0
        self.failed_requests = 0
        self.node_requests: Dict[str, int] = {}
    
    async def get_best_node(
        self, 
        strategy: str = "least_load",
        exclude_nodes: Optional[Set[str]] = None
    ) -> Optional[NodeInfo]:
        """Получение лучшей ноды по стратегии"""
        nodes = await self.service_registry.get_active_nodes()
        
        if not nodes:
            return None
        
        # Исключаем ноды если нужно
        if exclude_nodes:
            nodes = [node for node in nodes if node.node_id not in exclude_nodes]
        
        if not nodes:
            return None
        
        self.total_requests += 1
        
        # Выбираем ноду по стратегии
        if strategy == "round_robin":
            selected = nodes[self.request_count % len(nodes)]
            self.request_count += 1
        
        elif strategy == "least_load":
            selected = min(nodes, key=lambda n: n.load_score)
        
        elif strategy == "random":
            import random
            selected = random.choice(nodes)
        
        elif strategy == "least_connections":
            # Выбираем ноду с наименьшим количеством запросов
            selected = min(nodes, key=lambda n: self.node_requests.get(n.node_id, 0))
        
        else:
            # По умолчанию - round robin
            selected = nodes[self.request_count % len(nodes)]
            self.request_count += 1
        
        # Обновляем статистику
        self.node_requests[selected.node_id] = self.node_requests.get(selected.node_id, 0) + 1
        
        return selected
    
    def record_request_failure(self, node_id: str) -> None:
        """Запись неудачного запроса"""
        self.failed_requests += 1
        if node_id in self.node_requests and self.node_requests[node_id] > 0:
            self.node_requests[node_id] -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика load balancer'а"""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                (self.total_requests - self.failed_requests) / self.total_requests
                if self.total_requests > 0 else 1.0
            ),
            "node_requests": self.node_requests.copy(),
            "current_request_count": self.request_count
        }

class DistributedLock:
    """Распределенная блокировка через Redis"""
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis],
        key: str,
        timeout: float = 30.0,
        retry_delay: float = 0.1
    ):
        self.redis_client = redis_client
        self.key = f"lock:{key}"
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.lock_id = str(uuid.uuid4())
        self._locked = False
    
    async def __aenter__(self):
        """Асинхронное получение блокировки"""
        if not self.redis_client:
            # Fallback - всегда успешно без реального лока
            return self
        
        end_time = asyncio.get_event_loop().time() + self.timeout
        
        while asyncio.get_event_loop().time() < end_time:
            try:
                # Пытаемся установить блокировку
                result = await self.redis_client.set(
                    self.key,
                    self.lock_id,
                    nx=True,  # Only set if not exists
                    ex=int(self.timeout)  # Expire time
                )
                
                if result:
                    self._locked = True
                    return self
                
                # Ждем перед следующей попыткой
                await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                logger.warning(f"Failed to acquire distributed lock: {e}")
                break
        
        raise RuntimeError(f"Failed to acquire lock '{self.key}' within {self.timeout}s")
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Освобождение блокировки"""
        if self._locked and self.redis_client:
            try:
                # Удаляем блокировку только если это наша блокировка
                lua_script = """
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("del", KEYS[1])
                else
                    return 0
                end
                """
                await self.redis_client.eval(lua_script, 1, self.key, self.lock_id)
            except Exception as e:
                logger.warning(f"Failed to release distributed lock: {e}")
        
        self._locked = False

class SessionAffinity:
    """Session affinity для sticky sessions"""
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis],
        ttl: int = 3600  # 1 час
    ):
        self.redis_client = redis_client
        self.ttl = ttl
        
        # Локальный fallback
        self._local_sessions: Dict[str, str] = {}
        self._local_session_times: Dict[str, float] = {}
    
    async def get_node_for_session(self, session_id: str) -> Optional[str]:
        """Получение ноды для сессии"""
        if self.redis_client:
            try:
                node_id = await self.redis_client.get(f"session:{session_id}")
                return node_id
            except Exception as e:
                logger.warning(f"Failed to get session affinity: {e}")
        
        # Локальный fallback
        return self._local_sessions.get(session_id)
    
    async def set_node_for_session(self, session_id: str, node_id: str) -> None:
        """Установка ноды для сессии"""
        if self.redis_client:
            try:
                await self.redis_client.setex(f"session:{session_id}", self.ttl, node_id)
                return
            except Exception as e:
                logger.warning(f"Failed to set session affinity: {e}")
        
        # Локальный fallback
        import time
        self._local_sessions[session_id] = node_id
        self._local_session_times[session_id] = time.time()
        
        # Очистка старых сессий
        current_time = time.time()
        expired_sessions = [
            sid for sid, timestamp in self._local_session_times.items()
            if current_time - timestamp > self.ttl
        ]
        for sid in expired_sessions:
            self._local_sessions.pop(sid, None)
            self._local_session_times.pop(sid, None)
    
    async def remove_session(self, session_id: str) -> None:
        """Удаление привязки сессии"""
        if self.redis_client:
            try:
                await self.redis_client.delete(f"session:{session_id}")
                return
            except Exception as e:
                logger.warning(f"Failed to remove session affinity: {e}")
        
        # Локальный fallback
        self._local_sessions.pop(session_id, None)
        self._local_session_times.pop(session_id, None)

class ScalingManager:
    """Менеджер для управления горизонтальным масштабированием"""
    
    def __init__(
        self,
        service_registry: ServiceRegistry,
        load_balancer: LoadBalancer,
        session_affinity: Optional[SessionAffinity] = None
    ):
        self.service_registry = service_registry
        self.load_balancer = load_balancer
        self.session_affinity = session_affinity
        
        # Конфигурация
        self.auto_scaling_enabled = True
        self.max_load_threshold = 0.8  # 80% нагрузка
        self.min_nodes = 1
        self.max_nodes = 10
    
    async def route_request(
        self,
        session_id: Optional[str] = None,
        strategy: str = "least_load"
    ) -> Optional[NodeInfo]:
        """Маршрутизация запроса с учетом session affinity"""
        
        # Проверяем session affinity если включена
        if session_id and self.session_affinity:
            preferred_node_id = await self.session_affinity.get_node_for_session(session_id)
            if preferred_node_id:
                nodes = await self.service_registry.get_active_nodes()
                preferred_node = next(
                    (n for n in nodes if n.node_id == preferred_node_id),
                    None
                )
                if preferred_node and preferred_node.is_healthy:
                    return preferred_node
        
        # Обычная балансировка нагрузки
        selected_node = await self.load_balancer.get_best_node(strategy=strategy)
        
        # Устанавливаем session affinity если нужно
        if session_id and selected_node and self.session_affinity:
            await self.session_affinity.set_node_for_session(session_id, selected_node.node_id)
        
        return selected_node
    
    async def should_scale_out(self) -> bool:
        """Проверка необходимости масштабирования"""
        if not self.auto_scaling_enabled:
            return False
        
        nodes = await self.service_registry.get_active_nodes()
        
        if len(nodes) >= self.max_nodes:
            return False
        
        # Проверяем среднюю нагрузку
        if nodes:
            avg_load = sum(node.load_score for node in nodes) / len(nodes)
            return avg_load > self.max_load_threshold
        
        return len(nodes) < self.min_nodes
    
    async def should_scale_in(self) -> bool:
        """Проверка возможности уменьшения масштаба"""
        if not self.auto_scaling_enabled:
            return False
        
        nodes = await self.service_registry.get_active_nodes()
        
        if len(nodes) <= self.min_nodes:
            return False
        
        # Проверяем что все ноды недогружены
        if nodes:
            max_load = max(node.load_score for node in nodes)
            return max_load < (self.max_load_threshold * 0.3)  # Менее 30% от порога
        
        return False
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Статус кластера"""
        nodes = await self.service_registry.get_active_nodes()
        
        if not nodes:
            return {
                "status": "no_nodes",
                "total_nodes": 0,
                "healthy_nodes": 0,
                "average_load": 0.0,
                "load_balancer_stats": self.load_balancer.get_stats()
            }
        
        healthy_nodes = [n for n in nodes if n.is_healthy]
        avg_load = sum(node.load_score for node in healthy_nodes) / len(healthy_nodes) if healthy_nodes else 0
        
        status = "healthy"
        if not healthy_nodes:
            status = "unhealthy"
        elif len(healthy_nodes) < len(nodes) * 0.5:  # Менее 50% здоровых нод
            status = "degraded"
        elif avg_load > self.max_load_threshold:
            status = "overloaded"
        
        return {
            "status": status,
            "total_nodes": len(nodes),
            "healthy_nodes": len(healthy_nodes),
            "average_load": avg_load,
            "nodes": [node.to_dict() for node in nodes],
            "load_balancer_stats": self.load_balancer.get_stats(),
            "scaling": {
                "should_scale_out": await self.should_scale_out(),
                "should_scale_in": await self.should_scale_in(),
                "auto_scaling_enabled": self.auto_scaling_enabled,
                "min_nodes": self.min_nodes,
                "max_nodes": self.max_nodes
            }
        }






