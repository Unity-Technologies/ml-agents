import enlighten


class ProgressBar:
    manager = enlighten.get_manager()

    def __init__(self, name: str, unit: str, count: int, total: int):
        self.progress_bar = ProgressBar.manager.counter(
            desc=name, unit=unit, count=count, total=total
        )
        self.progress_bar.refresh()

    def update(self, count: int) -> None:
        """
        Updates the progress bar based on the current count.
        :param count: Current count.
        """
        self.progress_bar.count = count
        self.progress_bar.refresh()
