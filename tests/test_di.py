from threading import Thread

import pytest

from openaivec._di import CircularDependencyError, Container, ProviderError


class ServiceA:
    def __init__(self):
        self.value = "A"


class ServiceB:
    def __init__(self, service_a: ServiceA):
        self.service_a = service_a
        self.value = "B"


class ServiceC:
    def __init__(self):
        self.counter = 0

    def increment(self):
        self.counter += 1
        return self.counter


class ServiceD:
    def __init__(self, service_e):
        self.service_e = service_e
        self.value = "D"


class ServiceE:
    def __init__(self, service_d):
        self.service_d = service_d
        self.value = "E"


class TestContainer:
    def test_singleton_registration_and_resolution(self):
        container = Container()
        container.register(ServiceA, lambda: ServiceA())

        instance1 = container.resolve(ServiceA)
        instance2 = container.resolve(ServiceA)

        assert instance1 is instance2  # Always singleton
        assert instance1.value == "A"

    def test_dependency_injection(self):
        container = Container()
        container.register(ServiceA, lambda: ServiceA())
        container.register(ServiceB, lambda: ServiceB(container.resolve(ServiceA)))

        service_b = container.resolve(ServiceB)
        assert service_b.value == "B"
        assert service_b.service_a.value == "A"

    def test_circular_dependency_detection(self):
        container = Container()

        # Create circular dependency: D -> E -> D
        container.register(ServiceD, lambda: ServiceD(container.resolve(ServiceE)))
        container.register(ServiceE, lambda: ServiceE(container.resolve(ServiceD)))

        with pytest.raises(CircularDependencyError) as exc_info:
            container.resolve(ServiceD)

        assert "Circular dependency detected" in str(exc_info.value)

    def test_provider_error_handling(self):
        container = Container()

        def failing_provider():
            raise ValueError("Provider failed")

        container.register(ServiceA, failing_provider)

        with pytest.raises(ProviderError) as exc_info:
            container.resolve(ServiceA)

        assert "Failed to create instance of ServiceA" in str(exc_info.value)

    def test_unregistered_class_resolution(self):
        container = Container()

        with pytest.raises(ValueError) as exc_info:
            container.resolve(ServiceA)

        assert "No provider registered for class ServiceA" in str(exc_info.value)

    def test_registration_overwrite(self):
        """Test that registration can be overwritten."""
        container = Container()

        # First registration
        container.register(ServiceA, lambda: ServiceA())
        instance1 = container.resolve(ServiceA)
        assert instance1.value == "A"

        # Overwrite registration with a different provider
        def custom_provider():
            service = ServiceA()
            service.value = "Modified A"
            return service

        container.register(ServiceA, custom_provider)

        # Should get a new instance with the new provider
        instance2 = container.resolve(ServiceA)
        assert instance2.value == "Modified A"
        assert instance1 is not instance2  # Different instances

    def test_is_registered(self):
        container = Container()

        assert not container.is_registered(ServiceA)

        container.register(ServiceA, lambda: ServiceA())

        assert container.is_registered(ServiceA)

    def test_unregister(self):
        container = Container()
        container.register(ServiceA, lambda: ServiceA())

        assert container.is_registered(ServiceA)

        container.unregister(ServiceA)

        assert not container.is_registered(ServiceA)

    def test_unregister_unregistered_class(self):
        container = Container()

        with pytest.raises(ValueError) as exc_info:
            container.unregister(ServiceA)

        assert "Class ServiceA is not registered" in str(exc_info.value)

    def test_clear(self):
        container = Container()
        container.register(ServiceA, lambda: ServiceA())
        container.register(ServiceC, lambda: ServiceC())

        # Resolve to create instances
        container.resolve(ServiceA)
        container.resolve(ServiceC)

        assert container.is_registered(ServiceA)
        assert container.is_registered(ServiceC)

        container.clear()

        assert not container.is_registered(ServiceA)
        assert not container.is_registered(ServiceC)

    def test_thread_safety(self):
        """Test that the container is thread-safe."""
        container = Container()
        container.register(ServiceC, lambda: ServiceC())

        results: list[ServiceC] = []

        def resolve_and_increment():
            service = container.resolve(ServiceC)
            service.increment()
            results.append(service)

        threads = [Thread(target=resolve_and_increment) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All threads should get the same singleton instance
        assert len(set(id(service) for service in results)) == 1
        # Counter should be incremented 10 times
        assert results[0].counter == 10

    def test_multiple_containers_are_isolated(self):
        """Test that different container instances don't share state."""
        container1 = Container()
        container2 = Container()

        container1.register(ServiceA, lambda: ServiceA())

        assert container1.is_registered(ServiceA)
        assert not container2.is_registered(ServiceA)

        # Instances should be different
        service1 = container1.resolve(ServiceA)

        container2.register(ServiceA, lambda: ServiceA())
        service2 = container2.resolve(ServiceA)

        assert service1 is not service2

    def test_register_instance(self):
        """Test registering a pre-created instance."""
        container = Container()

        # Create a pre-configured instance
        service_instance = ServiceA()
        service_instance.value = "Pre-configured"

        # Register the instance directly
        container.register_instance(ServiceA, service_instance)

        # Resolve should return the same instance
        resolved = container.resolve(ServiceA)
        assert resolved is service_instance
        assert resolved.value == "Pre-configured"

        # Multiple resolves should return the same instance
        resolved2 = container.resolve(ServiceA)
        assert resolved2 is service_instance

    def test_register_instance_overwrite(self):
        """Test that register_instance can overwrite existing registrations."""
        container = Container()

        # First instance registration
        service_instance1 = ServiceA()
        service_instance1.value = "First instance"
        container.register_instance(ServiceA, service_instance1)

        resolved1 = container.resolve(ServiceA)
        assert resolved1 is service_instance1
        assert resolved1.value == "First instance"

        # Overwrite with a new instance
        service_instance2 = ServiceA()
        service_instance2.value = "Second instance"
        container.register_instance(ServiceA, service_instance2)

        # Should get the new instance
        resolved2 = container.resolve(ServiceA)
        assert resolved2 is service_instance2
        assert resolved2.value == "Second instance"
        assert resolved2 is not service_instance1

    def test_cross_registration_overwrite(self):
        """Test that register and register_instance can overwrite each other."""
        container = Container()

        # Start with regular registration
        container.register(ServiceA, lambda: ServiceA())
        instance1 = container.resolve(ServiceA)
        assert instance1.value == "A"

        # Overwrite with instance registration
        service_instance = ServiceA()
        service_instance.value = "Instance A"
        container.register_instance(ServiceA, service_instance)

        resolved = container.resolve(ServiceA)
        assert resolved is service_instance
        assert resolved.value == "Instance A"

        # Overwrite back with regular registration
        def new_provider():
            service = ServiceA()
            service.value = "New Provider A"
            return service

        container.register(ServiceA, new_provider)

        instance2 = container.resolve(ServiceA)
        assert instance2.value == "New Provider A"
        assert instance2 is not service_instance

    def test_register_instance_with_dependency_injection(self):
        """Test using register_instance in dependency injection scenarios."""
        container = Container()

        # Create and register a pre-configured database service
        db_service = ServiceA()
        db_service.value = "Production DB"
        container.register_instance(ServiceA, db_service)

        # Register a service that depends on the pre-configured instance
        container.register(ServiceB, lambda: ServiceB(container.resolve(ServiceA)))

        # Resolve the dependent service
        user_service = container.resolve(ServiceB)
        assert user_service.service_a is db_service
        assert user_service.service_a.value == "Production DB"

    def test_register_instance_thread_safety(self):
        """Test that register_instance is thread-safe."""
        container = Container()
        service_instance = ServiceC()

        container.register_instance(ServiceC, service_instance)

        results: list[ServiceC] = []

        def resolve_and_increment():
            service = container.resolve(ServiceC)
            service.increment()
            results.append(service)

        threads = [Thread(target=resolve_and_increment) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All threads should get the same registered instance
        assert len(set(id(service) for service in results)) == 1
        assert all(service is service_instance for service in results)
        # Counter should be incremented 5 times
        assert service_instance.counter == 5
