import click


class VnargsOption(click.Option):
    """Variadic Arguments Option:

    A custom Option type allowing for a variable number of arguments to be set
    as its value. By default, all args up to the next option (determined as a
    token prefixed by - or --) becomes the tuple value of this custom option.

    :param sep: the delimiter according which to separate the passed args
    :param min_nargs: integer specifying the minimum number args for the option
    :param max_nargs: integer specifying the maximum number args for the option
    """

    def __init__(self, *param_decls, separator=None,
                 min_nargs=1, max_nargs=float('inf'),
                 **attrs):

        # attrs specific to the VnargsOption subclass
        self.separator = separator
        self.min_nargs = min_nargs
        self.max_nargs = max_nargs

        nargs = attrs.pop('nargs', -1)
        assert nargs == -1, f'nargs, if set, must be -1 not {nargs}'

        super().__init__(*param_decls, **attrs)

    # Override click.Option's default add_to_parser method
    def add_to_parser(self, parser, ctx):

        # first add the Option instance using Click's standard add_to_parser
        click.Option.add_to_parser(self, parser, ctx)

        # custom func/method that hooks onto click.parser.Option.process
        def process_vnargs(value, state):
            vals = [value]
            while state.rargs:
                # condition(s) to stop adding arguments to this vnargs option
                # TODO: also break on getting max_nargs? (allows pos_arg after)
                reached_next_opt = _token_has_prefix(state.rargs[0],
                                                     click_parser.prefixes)
                if reached_next_opt:
                    break
                else:
                    vals.append(state.rargs.pop(0))

            # split the passed args appropriately if given valid separator
            # NOTE: both leading & trailing whitespaces on each arg are removed
            # NOTE: empty args (i.e. "--opt=a,,b") are ignored => {opt: (a, b)}
            if bool(self.separator):
                vals = [arg.strip() for arg in
                        ' '.join(vals).split(sep=self.separator)
                        if bool(arg)]

            # check the number of args is within the specified range
            if len(vals) < self.min_nargs or len(vals) > self.max_nargs:
                #  err_msg = f'option requires at least {self.min_nargs} args'
                #  raise click.BadOptionUsage(self.name, err_msg)
                # TODO: use Click's BadOptionUsage(UsageError) as in the above
                raise TypeError(f"option '{self.name}' takes between "
                                f"{self.min_nargs} to {self.max_nargs} args, "
                                f"given {len(vals)} args: {', '.join(vals)}")
                # TODO: choose join separator to ensure no conflict w/ self.sep

            # call click.parser.Option.process with the value being a
            # single tuple of all args passed to the vnargs option
            self.click_parser_process(tuple(vals), state)
            # END of process_vnargs function

        # get Click's default built-in parser for Options click.parser.Option
        opt = self.opts[0]
        click_parser = parser._short_opt.get(opt) or parser._long_opt.get(opt)

        # set aside Click's built-in parser.Option process method
        self.click_parser_process = click_parser.process

        # override the click.parser.Option instance's process method
        # w/ the custom process defined above
        click_parser.process = process_vnargs


# helper for determing if the next option has been reached in the parsing state
def _token_has_prefix(token, prefix_list):
    for pfx in prefix_list:
        if token.startswith(pfx):
            return True
    return False
