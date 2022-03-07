# Filter output of genhtml code coverate.
#
# awk variables passed in are
#
#    path      - path to index.html file, reflective of source directory.
#    fun_limit - percentage coverate limit to equal or beat to be excluded.
#    line_limit - percentage coverate limit to equal or beat to be excluded.
#
# At least one of fun_limit or line_limit must be met.
#
# Default for fun_limit and line_limit is 70.0%
#
BEGIN {
    tdcount= 0;
    path = gensub(/index.html$/, "", "g", path)

    if (fun_limit == "") {
	fun_limit = 70.0
    } else {
	fun_limit = fun_limit + 0.0
    }

    if (line_limit == "") {
	line_limit = 70.0
    } else {
	line_limit = line_limit + 0.0
    }
}

/<td class="coverFile">/ {
    if ($0 ~ /.gcov/) {
	file = gensub(/^.*href="(.*)\.gcov\.html.*$/, "\\1", "g", $0)
	tdcount = 1
    } else {
	tdcount = 0
    }

    next
}

/<td class="cover(Per|Num)(Lo|Hi)">/ {
    if (tdcount > 0) {
	value = gensub(/^.*>(.*)<.*$/, "\\1", "g", $0)
	value = gensub("&nbsp;%", "", "g", value)
	item[tdcount] = value
	tdcount = (tdcount + 1) % 7
	if ((tdcount == 0) && (path !~ /.\/testing\//)) {
	    if (((item[3] + 0.0) < fun_limit) || ((item[1] + 0.0) < line_limit)) {
		printf "%s%s: ", path, file
		printf "%s %s %s %s %s %s\n",  item[1], item[2], item[3], item[4], item[5], item[6]
	    }
	}
    }
}
