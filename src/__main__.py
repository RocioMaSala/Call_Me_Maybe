from .call_me_maybe import main

try:
    main()
except KeyboardInterrupt:
    print("\nHey! You are the problem :( ", end="")
