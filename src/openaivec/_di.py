from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, TypeVar

__all__ = []

"""Simple dependency injection container with singleton lifecycle management.

This module provides a lightweight dependency injection container that manages
service instances with singleton lifecycle semantics. All registered services
are created once and reused across multiple resolve calls.

Example:
    ```python
    from openaivec.di import Container

    class DatabaseService:
        def __init__(self):
            self.connection = "database://localhost"

    container = Container()
    container.register(DatabaseService, lambda: DatabaseService())

    db1 = container.resolve(DatabaseService)
    db2 = container.resolve(DatabaseService)
    print(db1 is db2)  # True - same instance
    ```
"""

__all__ = ["Container", "Provider", "CircularDependencyError", "ProviderError"]

T = TypeVar("T")
Provider = Callable[[], T]


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected during service resolution.

    This exception is thrown when the container detects that service resolution
    would result in an infinite loop due to circular dependencies between services.

    Attributes:
        message: A descriptive error message including the dependency chain.

    Example:
        ```python
        # ServiceA depends on ServiceB, ServiceB depends on ServiceA
        try:
            container.resolve(ServiceA)
        except CircularDependencyError as e:
            print(f"Circular dependency: {e}")
        ```
    """

    pass


class ProviderError(Exception):
    """Raised when a provider function fails to create a service instance.

    This exception wraps any underlying exception that occurs during service
    instantiation, providing additional context about which service failed.

    Attributes:
        message: A descriptive error message including the service name.

    Example:
        ```python
        def failing_provider():
            raise ValueError("Database connection failed")
        container.register(DatabaseService, failing_provider)
        try:
            container.resolve(DatabaseService)
        except ProviderError as e:
            print(f"Service creation failed: {e}")
        ```
    """

    pass


@dataclass
class Container:
    """A simple dependency injection container with singleton lifecycle and thread safety.

    This container manages service registration and resolution with automatic singleton
    lifecycle management. All registered services are created once and cached for
    subsequent resolves. The container is thread-safe and detects circular dependencies.

    Attributes:
        _instances: Cache of created singleton instances.
        _providers: Registry of provider functions for each service type.
        _lock: Thread synchronization lock for safe concurrent access.
        _resolving: Set of services currently being resolved (for cycle detection).

    Example:
        Basic container usage:
        ```python
        container = Container()
        container.register(str, lambda: "Hello, World!")
        message = container.resolve(str)
        print(message)  # Hello, World!
        ```

        Dependency injection:
        ```python
        class DatabaseService:
            def __init__(self):
                self.connected = True

        class UserService:
            def __init__(self, db: DatabaseService):
                self.db = db

        container.register(DatabaseService, lambda: DatabaseService())
        container.register(UserService, lambda: UserService(container.resolve(DatabaseService)))
        user_service = container.resolve(UserService)
        print(user_service.db.connected)  # True
        ```
    """

    _instances: dict[type[Any], Any] = field(default_factory=dict)
    _providers: dict[type[Any], Provider[Any]] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)
    _resolving: set[type[Any]] = field(default_factory=set)

    def register(self, cls: type[T], provider: Provider[T]) -> None:
        """Register a provider function for a service type.

        The provider function will be called once to create the singleton instance
        when the service is first resolved. Subsequent resolves will return the
        cached instance. If the service is already registered, it will be overwritten.

        Args:
            cls: The service class/type to register.
            provider: A callable that creates and returns an instance of the service.

        Example:
            ```python
            container = Container()
            container.register(str, lambda: "Hello")
            container.register(int, lambda: 42)
            # Overwrite existing registration
            container.register(str, lambda: "World")
            ```
        """
        with self._lock:
            if cls in self._providers and cls in self._instances:
                del self._instances[cls]

            self._providers[cls] = provider

    def register_instance(self, cls: type[T], instance: T) -> None:
        """Register a pre-created instance for a service type.

        The provided instance will be stored directly in the container and returned
        for all future resolve calls. This is useful when you have a pre-configured
        instance or when you want to register instances that require complex initialization.
        If the service is already registered, it will be overwritten.

        Args:
            cls: The service class/type to register.
            instance: The pre-created instance to register.

        Example:
            ```python
            container = Container()
            db_service = DatabaseService("production://db")
            container.register_instance(DatabaseService, db_service)
            resolved = container.resolve(DatabaseService)
            print(resolved is db_service)  # True - same instance
            # Overwrite existing registration
            new_db_service = DatabaseService("staging://db")
            container.register_instance(DatabaseService, new_db_service)
            ```
        """
        with self._lock:
            self._instances[cls] = instance
            self._providers[cls] = lambda: instance

    def resolve(self, cls: type[T]) -> T:
        """Resolve a service instance, creating it if necessary.

        Returns the singleton instance for the requested service type. If this is
        the first time the service is resolved, the provider function is called
        to create the instance, which is then cached for future resolves.

        Args:
            cls: The service class/type to resolve.

        Returns:
            The singleton instance of the requested service type.

        Raises:
            ValueError: If no provider is registered for the service type.
            CircularDependencyError: If a circular dependency is detected.
            ProviderError: If the provider function fails to create the instance.

        Example:
            ```python
            container = Container()
            container.register(str, lambda: "Hello, World!")
            message1 = container.resolve(str)
            message2 = container.resolve(str)
            print(message1 is message2)  # True - same instance
            ```
        """
        with self._lock:
            if cls in self._resolving:
                dependency_chain = " -> ".join(c.__name__ for c in self._resolving) + f" -> {cls.__name__}"
                raise CircularDependencyError(f"Circular dependency detected: {dependency_chain}")

            if cls not in self._providers:
                raise ValueError(f"No provider registered for class {cls.__name__}.")

            if cls in self._instances:
                return self._instances[cls]

            self._resolving.add(cls)

            try:
                instance = self._providers[cls]()

                self._instances[cls] = instance

                return instance

            except CircularDependencyError:
                raise
            except Exception as e:
                raise ProviderError(f"Failed to create instance of {cls.__name__}: {str(e)}") from e
            finally:
                self._resolving.discard(cls)

    def is_registered(self, cls: type[Any]) -> bool:
        """Check if a service type is registered in the container.

        Args:
            cls: The service class/type to check.

        Returns:
            True if the service type has a registered provider, False otherwise.

        Example:
            ```python
            container = Container()
            print(container.is_registered(str))  # False
            container.register(str, lambda: "Hello")
            print(container.is_registered(str))  # True
            ```
        """
        with self._lock:
            return cls in self._providers

    def unregister(self, cls: type[Any]) -> None:
        """Unregister a service type from the container.

        Removes the provider function and any cached singleton instance for
        the specified service type.

        Args:
            cls: The service class/type to unregister.

        Raises:
            ValueError: If the service type is not registered.

        Example:
            ```python
            container = Container()
            container.register(str, lambda: "Hello")
            container.unregister(str)
            print(container.is_registered(str))  # False
            ```
        """
        with self._lock:
            if cls not in self._providers:
                raise ValueError(f"Class {cls.__name__} is not registered.")

            del self._providers[cls]

            if cls in self._instances:
                del self._instances[cls]

    def clear(self) -> None:
        """Clear all registrations and cached instances from the container.

        Removes all registered providers and their associated singleton instances.
        After calling this method, the container will be empty and ready for
        new registrations.

        Example:
            ```python
            container = Container()
            container.register(str, lambda: "Hello")
            container.register(int, lambda: 42)
            container.clear()
            print(container.is_registered(str))  # False
            print(container.is_registered(int))  # False
            ```
        """
        with self._lock:
            self._providers.clear()
            self._instances.clear()
            self._resolving.clear()

    def clear_singletons(self) -> None:
        """Clear all cached singleton instances from the container.

        Removes all cached singleton instances while keeping the registered
        providers intact. After calling this method, the next resolve call
        for any service will create a new instance using the provider function.

        Example:
            ```python
            container = Container()
            container.register(str, lambda: "Hello")
            instance1 = container.resolve(str)
            container.clear_singletons()
            instance2 = container.resolve(str)
            print(instance1 is instance2)
            # False - different instances after clearing singletons
            ```
        """
        with self._lock:
            self._instances.clear()
