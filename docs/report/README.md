# The extended abstract

`main.tex` is the two-page technical report the submission form asks for. `./build.sh` renders it
to `main.pdf` and copies it to `docs/TECHNICAL_REPORT.pdf`.

The limit is **two pages of main text**; references and appendices do not count. `build.sh`
checks that directly — it looks for body words past page two rather than counting pages, so a
report whose references spill onto a third page still passes and one whose conclusion spills does
not.

`nvidiaextabstract.cls`, `nvidiatechreport.cls` and `natbib.sty` are NVIDIA's own template files,
distributed to challenge participants as `V2D_Challenge_Extended_Abstract_Template.zip` and
released under **CC BY-SA 4.0**. They are unmodified. Everything else here is ours and is covered
by the repository's Apache-2.0 `LICENSE`.

This machine's TeX Live is missing `lmodern` and `zlmtt`, which the class requires. They were
unpacked into `~/texmf-extra` and registered with `updmap-user` rather than installed, so the
build needs no root:

    apt-get download lmodern && dpkg -x lmodern_*.deb ~/texmf-extra
    curl -L https://mirrors.ctan.org/fonts/zlmtt.zip -o zlmtt.zip   # unzip tex/zlmtt.sty into the tree
    TEXMFHOME=$HOME/texmf-extra/usr/share/texmf updmap-user --enable Map=lm.map

Without the font map, `pdflatex` fails with "auto expansion is only possible with scalable
fonts", which names neither `lmodern` nor the map.
