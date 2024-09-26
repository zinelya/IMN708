def add_verbose_arg(parser):
    parser.add_argument('-v', default="WARNING", const='INFO', nargs='?',
                        choices=['DEBUG', 'INFO', 'WARNING'], dest='verbose',
                        help='Produces verbose output depending on '
                             'the provided level. \nDefault level is warning, '
<<<<<<< HEAD
                             'default when using -v is info.')
=======
                             'default when using -v is info.')

>>>>>>> 00a804f0f49e53ed5663e0b9e359b8151bb65107
