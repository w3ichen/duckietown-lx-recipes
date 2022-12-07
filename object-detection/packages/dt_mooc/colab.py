from IPython.display import HTML, display


class ColabProgressBar:
    def __init__(self):
        self._pbar = display(ColabProgressBar._progress(0, 100), display_id=True)

    def update(self, progress: int):
        if self._pbar is not None:
            self._pbar.update(self._progress(progress))

    def transfer_monitor(self, handler):
        self.update(handler.progress.percentage)

    @staticmethod
    def _progress(value, max=100):
        return HTML(
            """
            <progress
                value='{value}'
                max='{max}',
                style='width: 100%'
            >
                {value}
            </progress>
        """.format(
                value=value, max=max
            )
        )
