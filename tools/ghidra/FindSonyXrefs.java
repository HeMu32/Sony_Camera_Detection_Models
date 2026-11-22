// Report references to selected addresses and their containing functions.
// Usage: -postScript FindSonyXrefs.java <address,...>
// @category SonyNrAnalysis
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.symbol.Reference;

public class FindSonyXrefs extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length == 0) {
            throw new IllegalArgumentException("usage: address,...");
        }
        for (String item : args[0].split(",")) {
            Address target = toAddr(Long.decode(item));
            println("TARGET " + target);
            int count = 0;
            for (Reference ref : getReferencesTo(target)) {
                Function owner = getFunctionContaining(ref.getFromAddress());
                String ownerText = owner == null
                        ? "(no function)"
                        : owner.getEntryPoint() + ":" + owner.getName();
                println("  from " + ref.getFromAddress() + " type=" +
                        ref.getReferenceType() + " owner=" + ownerText);
                count++;
            }
            println("COUNT " + count);
        }
    }
}
