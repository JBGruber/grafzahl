.is_windows <- function() {
    Sys.info()[['sysname']] == "Windows"
}

.gen_conda_path <- function(envvar = "GRAFZAHL_MINICONDA_PATH", bin = FALSE) {
    if (Sys.getenv(envvar) == "") {
        main_path <- reticulate::miniconda_path()
    } else {
        main_path <- Sys.getenv(envvar)
    }
    if (isFALSE(bin)) {
        return(main_path)
    }
    if (.is_windows()) {
        return(file.path(main_path, "Scripts", "conda.exe"))
    }
    file.path(main_path, "bin", "conda")
}

## list all conda envs, but restrict to .gen_conda_path
## Should err somehow
.list_condaenvs <- function() {
    all_condaenvs <- reticulate::conda_list(conda = .gen_conda_path(bin = TRUE))
    ##if (.is_windows()) {
    return(all_condaenvs$name)
    ##}
    ##all_condaenvs[grepl(.gen_conda_path(), all_condaenvs$python),]$name
}

.have_conda <- function() {
    ## !is.null(tryCatch(reticulate::conda_list(), error = function(e) NULL))
    ## Not a very robust test, but take it.
    file.exists(.gen_conda_path(bin = TRUE))
}

## check if a vector of packages is installed in the virtual-env
.py_check_installed <- function(x, envname) {
  if (is.null(x)) return(FALSE)
  if (nchar(Sys.getenv("RETICULATE_PYTHON")) > 0) {
    return(x %in% reticulate::py_list_packages()$package)
  } else {
    return(x %in% trimws(reticulate::py_list_packages(
      Sys.getenv("GRAFZAHL_PYTHON_PATH", unset = envname))$package
    ))
  }
}


#' @rdname detect_cuda
#' @export
detect_conda <- function() {
    if(!.have_conda()) {
        return(FALSE)
    }
    envnames <- grep("^grafzahl_condaenv", .list_condaenvs(), value = TRUE)
    # TODO: add py env
    length(envnames) != 0
}

#' @rdname detect_cuda
#' @export
detect_pyenv <- function() {
  envnames <- grep("^r-grafzahl", reticulate::virtualenv_list(), value = TRUE)
  length(envnames) != 0
}

.gen_envname <- function(cuda = TRUE, py = FALSE) {
    if (py) {
      envname <- "r-grafzahl"
    } else {
      envname <- "grafzahl_condaenv"
    }
    if (cuda) {
        envname <- paste0(envname, "_cuda")
    }
    return(envname)
}

.initialize_conda <- function(envname, verbose = FALSE) {
    if (is.null(getOption('python_init'))) {
        if (.is_windows()) {
            python_executable <- file.path(.gen_conda_path(), "envs", envname, "python.exe")
        } else {
            python_executable <- file.path(.gen_conda_path(), "envs", envname, "bin", "python")
        }
        ## Until rstydio/reticulate#1308 is fixed; mask it for now
        Sys.setenv(RETICULATE_MINICONDA_PATH = .gen_conda_path())
        reticulate::use_miniconda(python_executable, required = TRUE)
        options('python_init' = TRUE)
        if (verbose) {
            message("Conda environment ", envname, " is initialized.")
        }
    }
    return(invisible(NULL))
}

#' Detecting Miniconda And Cuda
#'
#' These functions detects miniconda and cuda.
#'
#' `detect_conda` conducts a test to check whether 1) a miniconda installation and 2) the grafzahl miniconda environment exist.
#'
#' `detect_cuda` checks whether cuda is available. If `setup_grafzahl` was executed with `cuda` being `FALSE`, this function will return `FALSE`. Even if `setup_grafzahl` was executed with `cuda` being `TRUE` but with any factor that can't enable cuda (e.g. no Nvidia GPU, the environment was incorrectly created), this function will also return `FALSE`.
#' @return boolean, whether the system is available.
#' @export
detect_cuda <- function() {
    options('python_init' = NULL)
    if (Sys.getenv("KILL_SWITCH") == "KILL") {
        return(NA)
    }
    envnames <- character()
    if (.have_conda()) envnames <- grep("^grafzahl_condaenv", .list_condaenvs(), value = TRUE)
    if (length(envnames) == 0) {
      envnames <- grep("^r-grafzahl", reticulate::virtualenv_list(), value = TRUE)
      if (length(envnames) == 0) {
        stop("No conda environment found. Run `setup_grafzahl` to bootstrap one.")
      }
      reticulate::use_virtualenv(Sys.getenv("GRAFZAHL_PYTHON_PATH", unset = envnames))
    } else {
      if (grepl("_cuda", envnames, fixed = TRUE)) {
        envname <- "grafzahl_condaenv_cuda"
      } else {
        envname <- "grafzahl_condaenv"
      }
      .initialize_conda(envname = envname, verbose = FALSE)
    }
    reticulate::source_python(system.file("python", "st.py", package = "grafzahl"))
    return(py_detect_cuda())
}

.install_gpu_pytorch <- function(cuda_version) {
    .initialize_conda(.gen_envname(cuda = TRUE))
    conda_executable <- .gen_conda_path(bin = TRUE)
    status <- system2(conda_executable, args = c("install", "-n", .gen_envname(cuda = TRUE), "pytorch", "pytorch-cuda", paste0("cudatoolkit=", cuda_version), "-c", "pytorch", "-c", "nvidia", "-y"))
    if (status != 0) {
        stop("Cannot set up `pytorch`.")
    }
    python_executable <- reticulate::py_config()$python
    status <- system2(python_executable, args = c("-m", "pip", "install", "simpletransformers==0.63.11", "\"transformers==4.30.2\"", "\"scipy==1.10.1\""))
    if (status != 0) {
        stop("Cannot set up `simpletransformers`.")
    }
}

#' Setup grafzahl
#'
#' Install a self-contained miniconda environment with all Python components (PyTorch, Transformers, Simpletransformers, etc) which grafzahl required. The default location is "~/.local/share/r-miniconda/envs/grafzahl_condaenv" (suffix "_cuda" is added if `cuda` is `TRUE`).
#' On Linux or Mac and if miniconda is not found, this function will also install miniconda. The path can be changed by the environment variable `GRAFZAHL_MINICONDA_PATH`
#' @param cuda logical, if `TRUE`, indicate whether a CUDA-enabled environment is wanted.
#' @param force logical, if `TRUE`, delete previous environment (if exists) and create a new environment
#' @param cuda_version character, indicate CUDA version, ignore if `cuda` is `FALSE`
#' @examples
#' # setup an environment with cuda enabled.
#' if (detect_conda() && interactive()) {
#'     setup_grafzahl(cuda = TRUE)
#' }
#' @return TRUE (invisibly) if installation is successful.
#' @export
setup_grafzahl <- function(cuda = FALSE, force = FALSE, cuda_version = "11.3", use_conda = TRUE) {
    envname <- .gen_envname(cuda = cuda, py = !use_conda)
    if (use_conda) {
      setup_grafzahl_conda(envname = envname, cuda = cuda, force = force, cuda_version = cuda_version)
    } else {
      setup_grafzahl_pyenv(envname = envname, cuda = cuda, force = force, cuda_version = cuda_version)
    }

}

setup_grafzahl_conda <- function(envname, cuda, force, cuda_version) {
  if (!.have_conda()) {
    if (!force) {
      message("No conda was found in ", .gen_conda_path())
      ans <- utils::menu(c("No", "Yes"), title = paste0("Do you want to install miniconda in ", .gen_conda_path()))
      if (ans == 1) {
        stop("Setup aborted.\n")
      }
    }
    reticulate::install_miniconda(.gen_conda_path(bin = FALSE), update = TRUE, force = TRUE)
  }
  allenvs <- .list_condaenvs()
  if (envname %in% allenvs && !force) {
    stop(paste0("Conda environment ", envname, " already exists.\nForce reinstallation by setting `force` to `TRUE`.\n"))
  }
  if (envname %in% allenvs && force) {
    reticulate::conda_remove(envname = envname, conda = .gen_conda_path(bin = TRUE))
  }
  ## The actual installation
  ## https://github.com/rstudio/reticulate/issues/779
  ##conda_executable <- file.path(.gen_conda_path(), "bin/conda")
  if (isTRUE(cuda)) {
    yml_file <- "grafzahl_gpu.yml"
  } else {
    yml_file <- "grafzahl.yml"
  }
  status <- system2(.gen_conda_path(bin = TRUE), args = c("env", "create",  paste0("-f=", system.file(yml_file, package = 'grafzahl')), "-n", envname))
  if (status != 0) {
    stop("Cannot set up the basic conda environment.")
  }
  if (isTRUE(cuda)) {
    .install_gpu_pytorch(cuda_version = cuda_version)
  }
  ## Post-setup checks
  if (!detect_conda()) {
    stop("Conda can't be detected.")
  }
  if (cuda) {
    if (!detect_cuda()) stop("Cuda wasn't configurated correctly.")
  }
  return(invisible(envname))
}


setup_grafzahl_pyenv <- function(envname, cuda, force, cuda_version) {
  if (nchar(Sys.getenv("RETICULATE_PYTHON")) > 0) {
    message("You provided a custom RETICULATE_PYTHON, so we assume you know what you ",
            "are doing managing your virtual environments. Good luck!")
  } else if (!reticulate::virtualenv_exists(Sys.getenv("GRAFZAHL_PYTHON_PATH", unset = envname))) {
    # this has turned out to be the easiest way to test if a suitable Python
    # version is present. All other methods load Python, which creates
    # some headache.
    t <- try(reticulate::virtualenv_create(Sys.getenv("GRAFZAHL_PYTHON_PATH", unset = envname)), silent = TRUE)
    if (methods::is(t, "try-error")) {
      permission <- TRUE
      if (!force) {
        permission <- utils::askYesNo(paste0(
          "No suitable Python installation was found on your system. ",
          "Do you want to run `reticulate::install_python()` to install it?"
        ))
      }

      if (permission) {
        if (utils::packageVersion("reticulate") < "1.19")
          stop("Your version of reticulate is too old for this action. Please update")
        
        if (Sys.which("git") == "") {}
          stop("Installation of this pythin version needs git. If you don't know how to get it, look here:",
               "https://happygitwithr.com/install-git.html")
        python <- reticulate::install_python()
        reticulate::virtualenv_create(Sys.getenv("GRAFZAHL_PYTHON_PATH", unset = envname),
                                      python = python)
      } else {
        stop("Aborted by user")
      }
    }
    reticulate::use_virtualenv(Sys.getenv("GRAFZAHL_PYTHON_PATH", unset = envname))
  } else {
    reticulate::use_virtualenv(Sys.getenv("GRAFZAHL_PYTHON_PATH", unset = envname))
  }
  # instrestingly, the specific packages for cuda are not needed at all using pip
  pkgs <- c("simpletransformers", "transformers", "scipy", "torch", "pandas", "tqdm", "emoji==0.6.0")

  if (all(.py_check_installed(pkgs, envname)) && !force) {
    warning("Skipping installation. Use `force` to force installation or update.")
    return(invisible(envname))
  }

  reticulate::py_install(pkgs, envname = Sys.getenv("GRAFZAHL_PYTHON_PATH", unset = envname))

  ## Post-setup checks
  # detect_conda: not necessary because use_virtualenv fails if env is not created
  if (cuda) {
    if (!detect_cuda()) stop("Cuda wasn't configurated correctly.")
  }
  return(invisible(envname))
}
