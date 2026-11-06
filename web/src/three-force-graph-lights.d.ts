// `lights` is a real prop of <ForceGraph3D> that its own .d.ts does not declare.
//
// The shipped type surface puts `lights` only on the instance
// (`react-force-graph-3d/dist/react-force-graph-3d.d.ts:160-161`:
// `lights(lights: Light[]): ForceGraph3DInstance`) and not on `ForceGraphProps`
// (that interface opens at :52 and has no `lights` member), while the component
// applies EVERY prop generically (`react-force-graph-3d/dist/react-force-graph-3d.js:212`:
// `Object.keys(omit(props, [...methodNames, ...initPropNames])).filter(changed)
// .forEach(p => _call(p, props[p]))`, and `_call` at :202-208 is
// `comp[method].apply(comp, args)`). So `<ForceGraph3D lights={…}>` reaches
// `instance.lights(…)`, which is the documented setter — and `tsc` rejects the
// JSX with `Property 'lights' does not exist on type …ForceGraphProps` because
// the declaration, not the code, is missing the entry.
//
// This error was invisible until #3816: TypeScript reports one failed JSX
// attributes check per element, and `EntityGraph.tsx` already carried the
// `linkDirectionalParticleThreeObject` error, so the element never got as far as
// reporting this one. Fixing that prop is what made it print.
//
// Declaring it here is what makes `npm run build` green without touching the
// lighting: deleting `lights` from `EntityGraph.tsx` would fall back to the
// library's default lights, which the comment above that memo says is exactly the
// flattening it is avoiding. And a suppression comment or a cast here would make
// the build green while leaving the prop's VALUE unchecked, which is the thing
// #3816 clause 5 refuses — so the type below is written down instead. `Light[]` is
// strictly less permissive than the error it replaces: a non-Light array is now a
// type error where before the whole prop was untyped excess.
//
// Augmentation is possible because the package re-exports the interface
// (`export type { … ForceGraphProps … }`, :179). The type parameters, defaults
// included, must match the original declaration exactly or TypeScript refuses the
// merge.
import type { Light } from "three";
import "react-force-graph-3d";

declare module "react-force-graph-3d" {
  interface ForceGraphProps<NodeType = {}, LinkType = {}> {
    /** Scene lights, applied through the instance's `lights(Light[])` setter. */
    lights?: Light[];
  }
}
